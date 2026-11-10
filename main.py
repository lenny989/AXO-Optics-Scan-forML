"""Train the fully connected net. Every setting is a command-line option, and every
run is saved in its own folder together with its settings, so runs can be compared.

Step 0 (once):  python -c "from data import build_cache; build_cache('path/to/data', 'cache/xrd')"
Quick test:     python main.py --cache cache/xrd --val-scans run_B --epochs 2 --max-batches 20
Full run:       python main.py --cache cache/xrd --val-scans run_B run_C --epochs 500
Experiment:     python main.py ... --activation gelu --dropout 0.3 --hidden 2048 512 --name gelu_wide
Compare runs:   python compare.py
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import XrdDataset, make_loaders
from model import ACTIVATIONS, CNN, FCNet

LOSSES = {"huber": nn.SmoothL1Loss, "mse": nn.MSELoss, "l1": nn.L1Loss}


def get_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--cache", required=True, help="cache prefix, e.g. cache/xrd")
    p.add_argument("--val-scans", nargs="+", required=True, help="run names held out for validation")
    p.add_argument("--exclude-scans", nargs="+", default=[],
                   help="drop every run whose name contains one of these texts, e.g. 236cm")
    p.add_argument("--no-augment", action="store_true", help="turn shift augmentation off")
    p.add_argument("--signal-frac", type=float, default=0.3, help="threshold that finds the signal")
    p.add_argument("--repeat", type=int, default=5, help="passes over the images per epoch")
    # model
    p.add_argument("--model", choices=["fc", "cnn"], default="fc")
    p.add_argument("--hidden", nargs="+", type=int, default=[1024, 512, 128], help="fc: hidden layers")
    p.add_argument("--channels", nargs="+", type=int, default=[16, 32, 64, 128],
                   help="cnn: output channels of the conv blocks (each block halves the image)")
    p.add_argument("--head", nargs="+", type=int, default=[256], help="cnn: fully connected layers after the convs")
    p.add_argument("--pool", choices=["flatten", "avg"], default="flatten",
                   help="cnn: flatten keeps positions, avg averages them away")
    p.add_argument("--activation", choices=ACTIVATIONS, default="relu")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-batchnorm", action="store_true")
    # training
    p.add_argument("--loss", choices=LOSSES, default="huber")
    p.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    # bookkeeping
    p.add_argument("--name", default=None, help="run name (default: made from the settings)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="continue the run given by --name")
    p.add_argument("--max-batches", type=int, default=0, help="debug: stop each epoch early")
    return p.parse_args()


def auto_name(a):
    hidden = "-".join(map(str, a.hidden))
    return (f"{time.strftime('%m%d-%H%M')}_{a.activation}_{hidden}_do{a.dropout}_lr{a.lr}"
            + ("_noaug" if a.no_augment else ""))


def make_optimizer(a, params):
    if a.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    if a.optimizer == "adam":
        return torch.optim.Adam(params, lr=a.lr, weight_decay=a.weight_decay)
    return torch.optim.SGD(params, lr=a.lr, weight_decay=a.weight_decay, momentum=0.9)


@torch.no_grad()
def evaluate(model, loader, y_std, device, max_batches=0):
    """MAE per axis in REAL units (error in standardized space * y_std)."""
    model.eval()
    err, n = torch.zeros(2), 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        err += (model(x) - y).abs().sum(0).cpu()
        n += len(x)
    return (err / n) * y_std


@torch.no_grad()
def predict(model, loader, device, max_batches=0):
    """Predictions and targets (both standardized) in loader order. Evaluation mode: no dropout."""
    model.eval()
    preds, trues = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        preds.append(model(x.to(device)).cpu())
        trues.append(y)
    return torch.cat(preds), torch.cat(trues)


LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_mae_dx", "val_mae_dz",
              "train_mae_dx", "train_mae_dz", "lr", "seconds"]


def main():
    a = get_args()
    if a.resume and not a.name:
        raise SystemExit("--resume needs --name <existing run>")
    a.name = a.name or auto_name(a)
    out = Path(a.runs_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, stats = make_loaders(
        a.cache, a.val_scans, a.batch_size, a.repeat, a.workers,
        augment=not a.no_augment, signal_frac=a.signal_frac, exclude=a.exclude_scans)
    y_std = torch.tensor(stats["y_std"], dtype=torch.float32)
    if len(train_dl) == 0:
        raise SystemExit("fewer training samples than one batch: lower --batch-size or raise --repeat")

    # For the extra metrics: which run each validation image belongs to, its real targets, and a fixed
    # subset of training images that is evaluated WITHOUT augmentation (train MAE comparable to val MAE).
    val_ds, tr_ds = val_dl.dataset, train_dl.dataset
    val_run = np.load(f"{a.cache}_scans.npy")[val_ds.idx].astype(str)
    val_true_real = np.asarray(val_ds.targets)[val_ds.idx]
    y_mean = np.asarray(stats["y_mean"])
    sub = np.sort(np.random.default_rng(0).choice(tr_ds.idx, size=min(5000, len(tr_ds.idx)), replace=False))
    chk_workers = min(2, a.workers)
    chk_dl = DataLoader(XrdDataset(tr_ds.images_path, tr_ds.targets, sub, tr_ds.s, augment=False),
                        batch_size=a.batch_size * 2, num_workers=chk_workers,
                        persistent_workers=chk_workers > 0)

    # Baseline: always predict the training mean (= 0 after standardizing).
    base = torch.zeros(2)
    for _, y in val_dl:
        base += y.abs().sum(0)
    base = base / len(val_dl.dataset) * y_std

    in_shape = tuple(train_dl.dataset[0][0].shape)           # (1, 128, 128), (1, 256, 256), ... from the cache
    if a.model == "cnn":
        model = CNN(in_shape=in_shape, channels=tuple(a.channels), head=tuple(a.head), p_drop=a.dropout,
                    activation=a.activation, batchnorm=not a.no_batchnorm, pool=a.pool).to(device)
    else:
        model = FCNet(in_dim=int(np.prod(in_shape)), hidden=tuple(a.hidden), p_drop=a.dropout,
                      activation=a.activation, batchnorm=not a.no_batchnorm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"run={a.name}  device={device}  params={n_params / 1e6:.1f}M  "
          f"train batches/epoch={len(train_dl)}  val samples={len(val_dl.dataset)}")
    print(f"baseline MAE (predict the mean)  dx={base[0]:.4f}  dz={base[1]:.4f}")

    # Save every setting of this run.
    cfg = vars(a) | dict(n_params=n_params, baseline_dx=base[0].item(), baseline_dz=base[1].item())
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    opt = make_optimizer(a, model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs) if a.scheduler == "cosine" else None
    loss_fn = LOSSES[a.loss]()

    start, best, summary = 0, float("inf"), {}
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched:
            sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        if (out / "summary.json").exists():
            summary = json.loads((out / "summary.json").read_text())
        print(f"resumed at epoch {start}")

    fresh = start == 0
    log_file = out / "log.csv"
    need_header = fresh or not log_file.exists() or log_file.stat().st_size == 0
    # A resumed run keeps the columns its log already has (a run started with an older version of
    # this script simply gets no new columns), so the CSV never ends up with ragged rows.
    fields = LOG_FIELDS if need_header else log_file.read_text().splitlines()[0].split(",")
    log = open(log_file, "w" if need_header else "a", newline="")
    writer = csv.DictWriter(log, fieldnames=fields, extrasaction="ignore")
    if need_header:
        writer.writeheader()
    scan_file = out / "per_scan.csv"          # validation MAE of every held-out run, every epoch
    scan_new = fresh or not scan_file.exists()
    scan_log = open(scan_file, "w" if scan_new else "a", newline="")
    scan_writer = csv.writer(scan_log)
    if scan_new:
        scan_writer.writerow(["epoch", "scan", "n", "mae_dx", "mae_dz"])

    for epoch in range(start, a.epochs):
        t0 = time.time()
        model.train()
        total, nb = 0.0, 0
        for i, (x, y) in enumerate(train_dl):
            if a.max_batches and i >= a.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        lr_now = opt.param_groups[0]["lr"]
        if sched:
            sched.step()

        val = evaluate(model, val_dl, y_std, device, a.max_batches)
        mean_val = val.mean().item()

        # Extra metrics. They are optional: if something in here fails, training carries on without them.
        nan = float("nan")
        val_loss, tr_mae, preds = nan, torch.full((2,), nan), None
        try:
            pv, tv = predict(model, val_dl, device, a.max_batches)
            val_loss = loss_fn(pv, tv).item()
            pt, tt = predict(model, chk_dl, device, a.max_batches)
            tr_mae = (pt - tt).abs().mean(0) * y_std
            runs = val_run[:len(pv)]
            for s in np.unique(runs):
                m = torch.from_numpy(runs == s)
                e = (pv[m] - tv[m]).abs().mean(0) * y_std
                scan_writer.writerow([epoch, s, int(m.sum()), e[0].item(), e[1].item()])
            scan_log.flush()
            preds = (pv.numpy() * y_std.numpy() + y_mean, runs)
        except Exception as ex:
            print(f"WARNING: extra metrics failed this epoch: {ex!r}")

        writer.writerow(dict(epoch=epoch, train_loss=total / nb, val_loss=val_loss,
                             val_mae_dx=val[0].item(), val_mae_dz=val[1].item(),
                             train_mae_dx=tr_mae[0].item(), train_mae_dz=tr_mae[1].item(),
                             lr=lr_now, seconds=round(time.time() - t0, 1)))
        log.flush()
        print(f"epoch {epoch:3d}  loss {total / nb:.4f}  val loss {val_loss:.4f}  "
              f"val MAE dx={val[0]:.4f} dz={val[1]:.4f}  train MAE dx={tr_mae[0]:.4f} dz={tr_mae[1]:.4f}")

        if mean_val < best:
            best = mean_val
            torch.save(dict(model=model.state_dict(), stats=stats, epoch=epoch, config=cfg),
                       out / "best.pt")
            if preds is not None:                 # predictions of the best model, real units
                p, runs = preds
                n = len(p)
                pd.DataFrame(dict(scan=runs, index=val_ds.idx[:n],
                                  true_dx=val_true_real[:n, 0], true_dz=val_true_real[:n, 1],
                                  pred_dx=p[:, 0], pred_dz=p[:, 1], epoch=epoch)
                             ).to_csv(out / "predictions.csv", index=False)
            summary.update(best_val_mean=mean_val, best_val_dx=val[0].item(),
                           best_val_dz=val[1].item(), best_epoch=epoch)
        summary.update(last_epoch=epoch, last_train_loss=total / nb,
                       baseline_mean=float(base.mean()))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))

        # last.pt holds everything needed to resume after a crash or a Slurm time-out.
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict() if sched else None,
                        epoch=epoch, best=best, stats=stats), out / "last.pt")


if __name__ == "__main__":
    main()