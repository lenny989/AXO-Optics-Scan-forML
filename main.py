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
import torch
import torch.nn as nn

from data import make_loaders
from model import ACTIVATIONS, FCNet

LOSSES = {"huber": nn.SmoothL1Loss, "mse": nn.MSELoss, "l1": nn.L1Loss}


def get_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--cache", required=True, help="cache prefix, e.g. cache/xrd")
    p.add_argument("--val-scans", nargs="+", required=True, help="run names held out for validation")
    p.add_argument("--no-augment", action="store_true", help="turn shift augmentation off")
    p.add_argument("--signal-frac", type=float, default=0.3, help="threshold that finds the signal")
    p.add_argument("--repeat", type=int, default=5, help="passes over the images per epoch")
    # model
    p.add_argument("--hidden", nargs="+", type=int, default=[1024, 512, 128])
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
        augment=not a.no_augment, signal_frac=a.signal_frac)
    y_std = torch.tensor(stats["y_std"], dtype=torch.float32)
    if len(train_dl) == 0:
        raise SystemExit("fewer training samples than one batch: lower --batch-size or raise --repeat")

    # Baseline: always predict the training mean (= 0 after standardizing).
    base = torch.zeros(2)
    for _, y in val_dl:
        base += y.abs().sum(0)
    base = base / len(val_dl.dataset) * y_std

    model = FCNet(hidden=tuple(a.hidden), p_drop=a.dropout, activation=a.activation,
                  batchnorm=not a.no_batchnorm).to(device)
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
    log = open(out / "log.csv", "w" if fresh else "a", newline="")
    writer = csv.writer(log)
    if fresh:
        writer.writerow(["epoch", "train_loss", "val_mae_dx", "val_mae_dz", "lr", "seconds"])

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
        writer.writerow([epoch, total / nb, val[0].item(), val[1].item(),
                         lr_now, round(time.time() - t0, 1)])
        log.flush()
        print(f"epoch {epoch:3d}  loss {total / nb:.4f}  val MAE dx={val[0]:.4f} dz={val[1]:.4f}")

        if mean_val < best:
            best = mean_val
            torch.save(dict(model=model.state_dict(), stats=stats, epoch=epoch, config=cfg),
                       out / "best.pt")
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