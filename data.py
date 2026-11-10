"""Data pipeline for predicting (X, Z) offsets from PILATUS detector images.

Two stages:
  1. build_cache(): slow, run ONCE. Reads .cbf files, cleans them, saves one array.
  2. XrdDataset:    fast, used every epoch. Reads the cache, augments, normalizes.

Labels come straight from the xlab log files:
  <data_root>/<run_name>/image_0000001.cbf   <->   log_0000001.txt
  target = (hexapod x, hexapod z) read from the log with the same number.
  scan   = the run folder name (used for the train/val split).
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------
# Stage 1: build the cache (run once)
# --------------------------------------------------------------------------
def find_keep_masks(img):
    """Rows/cols that are completely masked (negative) are module gaps = black bars."""
    bad = img < 0
    keep_rows = ~bad.all(axis=1)
    keep_cols = ~bad.all(axis=0)
    return keep_rows, keep_cols


def preprocess(img, keep_rows, keep_cols, out_size=128):
    img = img.astype(np.float32)
    img[img < 0] = 0                                  # remaining bad pixels
    img = img[keep_rows][:, keep_cols]                # remove black bars
    t = torch.from_numpy(img)[None, None]
    t = F.adaptive_avg_pool2d(t, (out_size, out_size))  # bin to fixed size
    return torch.log1p(t[0, 0]).numpy()               # log AFTER binning


_HEX = re.compile(r"^hexapod\s+([xz])\s*=\s*(\S+)")   # not "motors x", not "hexapod pivot_x"


def parse_log(path):
    """Return (hexapod_x, hexapod_z) from one xlab log file."""
    vals = {}
    with open(path) as f:
        for line in f:
            m = _HEX.match(line)
            if m:
                vals[m.group(1)] = float(m.group(2))
    if "x" not in vals or "z" not in vals:
        raise ValueError(f"no 'hexapod x' / 'hexapod z' line in {path}")
    return vals["x"], vals["z"]


def collect_samples(data_root):
    """Pair every image_N.cbf with log_N.txt in each run folder."""
    rows = []
    for run_dir in sorted(p for p in Path(data_root).iterdir() if p.is_dir()):
        for cbf in sorted(run_dir.glob("image_*.cbf")):
            log = run_dir / cbf.name.replace("image_", "log_").replace(".cbf", ".txt")
            if not log.exists():
                print("no log for", cbf)
                continue
            x, z = parse_log(log)
            rows.append((str(cbf), x, z, run_dir.name))
    return pd.DataFrame(rows, columns=["file", "dx", "dz", "scan"])


def build_cache(data_root, out_prefix, out_size=128):
    import fabio                                      # pip install fabio
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)   # create cache/ if missing
    df = collect_samples(data_root)
    if df.empty:
        raise SystemExit(f"no image_*.cbf + log_*.txt pairs found directly inside run folders of {data_root}")
    print(len(df), "samples in", df.scan.nunique(), "runs")

    first = fabio.open(df.file.iloc[0]).data
    keep_rows, keep_cols = find_keep_masks(first)

    # Write straight into the .npy file on disk (memory-mapped) instead of holding all images in RAM:
    # 256x256 would need about 8 GB of RAM otherwise. The file gets its real name only when complete,
    # so a job that dies halfway never leaves a half-filled cache behind.
    final = f"{out_prefix}_images.npy"
    part = final + ".part"
    imgs = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16,
                                     shape=(len(df), out_size, out_size))
    for i, f in enumerate(df.file):
        imgs[i] = preprocess(fabio.open(f).data, keep_rows, keep_cols, out_size)
        if i % 5000 == 0:
            print(f"{i}/{len(df)}", flush=True)
    imgs.flush()
    del imgs
    os.replace(part, final)
    np.save(f"{out_prefix}_targets.npy", df[["dx", "dz"]].to_numpy(np.float32))
    np.save(f"{out_prefix}_scans.npy", df.scan.to_numpy().astype(str))  # plain string array, no pickle


# --------------------------------------------------------------------------
# Stage 2: dataset used for training
# --------------------------------------------------------------------------
def split_by_scan(scans, val_scans):
    """Split by scan ID so neighbouring (near-identical) images never straddle the split."""
    is_val = np.isin(scans, val_scans)
    return np.where(~is_val)[0], np.where(is_val)[0]


def compute_stats(images, targets, train_idx, n=5000, seed=0):
    """Mean/std from TRAIN data only."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(train_idx, size=min(n, len(train_idx)), replace=False)
    sub.sort()
    x = np.asarray(images[sub], dtype=np.float32)
    y = np.asarray(targets[train_idx])
    y_std = y.std(0)
    if (y_std < 1e-8).any():       # a target that never changes would divide by zero
        print("WARNING: a target is constant in the training runs, std =", y_std)
        y_std = np.where(y_std < 1e-8, 1.0, y_std)
    return dict(x_mean=float(x.mean()), x_std=float(x.std()),
                y_mean=y.mean(0), y_std=y_std)


def shift_image(img, dy, dx):
    """Shift with zero fill (no wrap-around). out[y+dy, x+dx] = img[y, x]."""
    H, W = img.shape
    out = np.zeros_like(img)
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        img[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


class XrdDataset(Dataset):
    def __init__(self, images_path, targets, idx, stats,
                 augment=False, repeat=1, signal_frac=0.3):
        # Only the PATH is stored. The memory map is opened inside each worker process.
        # (Windows starts workers by pickling the dataset; pickling an open memmap
        #  would copy all the pixels and crash.)
        self.images_path = str(images_path)
        self._images = None
        self.targets = targets
        self.idx = np.asarray(idx)
        self.s = stats
        self.augment = augment
        self.repeat = repeat          # repeat=5 -> ~5 differently shifted views per image
        self.signal_frac = signal_frac

    def __getstate__(self):                       # never pickle the open memmap
        state = self.__dict__.copy()
        state["_images"] = None
        return state

    def __len__(self):
        return len(self.idx) * self.repeat

    def __getitem__(self, i):
        if self._images is None:                  # first use in this process
            self._images = np.load(self.images_path, mmap_mode="r")
        j = self.idx[i % len(self.idx)]
        img = np.array(self._images[j], dtype=np.float32)  # copy out of the memmap
        y = self.targets[j]                                # label does NOT change when shifting

        if self.augment:
            # bounding box of the signal -> only allow shifts that keep it in frame
            ys, xs = np.nonzero(img > self.signal_frac * img.max())
            if len(ys):
                H, W = img.shape
                dy = np.random.randint(-ys.min(), H - ys.max())
                dx = np.random.randint(-xs.min(), W - xs.max())
                img = shift_image(img, dy, dx)

        img = (img - self.s["x_mean"]) / self.s["x_std"]
        y = (y - self.s["y_mean"]) / self.s["y_std"]
        return torch.from_numpy(img)[None], torch.from_numpy(y.astype(np.float32))


def make_loaders(prefix, val_scans, batch_size=256, repeat=5, num_workers=4,
                 augment=True, signal_frac=0.3, exclude=()):
    images_path = f"{prefix}_images.npy"
    images = np.load(images_path, mmap_mode="r")      # used here only for the statistics
    targets = np.load(f"{prefix}_targets.npy")
    scans = np.load(f"{prefix}_scans.npy")

    # Runs whose name contains any `exclude` pattern are dropped completely
    # (not in training, not in validation, not in the statistics).
    keep = np.ones(len(scans), dtype=bool)
    all_runs = sorted(str(s) for s in np.unique(scans))
    if exclude:
        for pat in exclude:
            if not any(pat in s for s in all_runs):
                raise ValueError(f"--exclude-scans pattern {pat!r} matches no run. Runs: {all_runs}")
        dropped = [s for s in all_runs if any(pat in s for pat in exclude)]
        keep = ~np.isin(scans, dropped)
        print(f"excluding {len(dropped)} run(s) with {(~keep).sum()} images: {dropped}")

    known = set(np.unique(scans[keep]))
    unknown = [s for s in val_scans if s not in known]
    if unknown:
        raise ValueError(f"unknown val scan(s) {unknown} (or excluded). Valid names: {sorted(known)}")
    if known <= set(val_scans):
        raise ValueError("all runs are in --val-scans; nothing left for training")
    tr, va = split_by_scan(scans, val_scans)
    tr, va = tr[keep[tr]], va[keep[va]]
    stats = compute_stats(images, targets, tr)

    train_ds = XrdDataset(images_path, targets, tr, stats, augment=augment, repeat=repeat,
                          signal_frac=signal_frac)
    val_ds = XrdDataset(images_path, targets, va, stats, augment=False)

    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    return (DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, batch_size * 2, **kw),
            stats)