"""Grouped comparison plots: one PNG per question, in runs/plots/.

    python plot_groups.py

Left panel : validation MAE per epoch (mean of dx and dz). Thin line = raw, thick = 5-epoch average.
Right panel: mean of the LAST 10 epochs +- standard deviation. This is less biased than the single
             best epoch, because a noisy run gets no extra chances to hit a lucky low.
Runs that do not exist yet are skipped (the script says which).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

B = "lr_1e-3"      # the reference run (default settings, learning rate 1e-3, seed 0)
WIDE = "2048-1024-256"

# group file name -> title, and a list of panels: (panel title, [(run folder, label), ...])
GROUPS = {
    "1_lr_batchnorm_on": ("Learning rate, batch norm ON", [
        ("Learning rate", [("lr_1e-4", "1e-4"), ("lr_3e-4", "3e-4"), ("lr_1e-3", "1e-3"),
                           ("lr_3e-3", "3e-3"), ("lr_1e-2", "1e-2")])]),
    "2_lr_batchnorm_off": ("Learning rate, batch norm OFF", [
        ("Learning rate", [("exp_nobn_1e-5", "1e-5"), ("exp_nobn_1e-4", "1e-4"),
                           ("exp_nobn_3e-4", "3e-4"), ("exp_nobn", "1e-3"),
                           ("exp_nobn_3e-3", "3e-3")])]),
    "3_regularization": ("Regularization", [
        ("Dropout", [("exp_do0", "0"), ("exp_do01", "0.1"), (B, "0.2 (default)"), ("exp_do04", "0.4")]),
        ("Weight decay", [("exp_wd0", "0"), (B, "1e-4 (default)"), ("exp_wd1e-2", "1e-2")])]),
    "4_architecture": ("Architecture", [
        ("Width", [("exp_narrow", "256-128"), ("exp_mid", "512-256-128"), (B, "1024-512-128 (default)"),
                   ("exp_wide", WIDE), ("exp_xwide", "4096-2048-512")]),
        ("Depth", [(B, "3 layers (default)"), ("exp_deep", "5 layers")])]),
    "5_activation_and_loss": ("Activation function and loss", [
        ("Activation", [(B, "relu (default)"), ("exp_leaky", "leaky_relu"), ("exp_gelu", "gelu"),
                        ("exp_silu", "silu"), ("exp_tanh", "tanh")]),
        ("Loss", [(B, "huber (default)"), ("exp_l1", "l1"), ("exp_mse", "mse")])]),
    "6_augmentation": ("Shift augmentation", [
        ("Augmentation", [("exp_noaug", "off"), ("exp_sf01", "signal-frac 0.1 (small shifts)"),
                          (B, "0.3 (default)"), ("exp_sf06", "0.6 (large shifts)")])]),
    "7_training_setup": ("Training setup", [
        ("Batch size", [("exp_bs128", "128"), (B, "256 (default)"), ("exp_bs512", "512")]),
        ("Learning-rate schedule", [(B, "cosine (default)"), ("exp_nosched", "constant")])]),
}

# Final candidates and the seed-to-seed noise: configuration -> runs (seed 0, 1, 2).
SEEDS = {
    "default": [B, "exp_base_s1", "exp_base_s2"],
    "dropout 0": ["exp_do0", "exp_do0_s1", "exp_do0_s2"],
    f"wide {WIDE}": ["exp_wide", "exp_wide_s1", "exp_wide_s2"],
    "dropout 0 + wide": ["exp_do0_wide", "exp_do0_wide_s1", "exp_do0_wide_s2"],
}

FOOT = ("Validation MAE = mean of dx and dz on the two held-out runs. "
        "Dots: mean of the last 10 epochs, bars: standard deviation over those epochs.")


def load(runs_dir, run):
    f = Path(runs_dir) / run / "log.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if d.empty:
        return None
    d["mae"] = (d.val_mae_dx + d.val_mae_dz) / 2
    return d


def last10(d):
    t = d.mae.tail(10)
    return t.mean(), t.std()


def panel(axc, axd, title, items, runs_dir, group, rows):
    colors = plt.get_cmap("tab10").colors
    found = []
    for i, (run, label) in enumerate(items):
        d = load(runs_dir, run)
        if d is None:
            print(f"  missing: {run}  ({label})")
            continue
        c = colors[i % 10]
        m, s = last10(d)
        axc.plot(d.epoch, d.mae, color=c, alpha=0.25, lw=0.8)
        axc.plot(d.epoch, d.mae.rolling(5, min_periods=1, center=True).mean(), color=c, lw=2, label=label)
        found.append((label, m, s, c))
        rows.append(dict(group=group, panel=title, label=label, run=run, epochs=len(d),
                         last10_mean=m, last10_std=s, best_epoch_value=d.mae.min()))
    axc.set_yscale("log")
    axc.set_title(title)
    axc.set_xlabel("epoch")
    axc.set_ylabel("validation MAE")
    axc.grid(alpha=0.3, which="both")
    if found:
        axc.legend(fontsize=8, loc="upper right")
    n = len(found)
    for k, (label, m, s, c) in enumerate(found):
        y = n - 1 - k
        axd.errorbar(m, y, xerr=s, fmt="o", color=c, capsize=3)
        axd.annotate(f"{m:.4f}", (m, y), xytext=(7, -3), textcoords="offset points", fontsize=8)
    axd.set_yticks(range(n))
    axd.set_yticklabels([f[0] for f in found][::-1], fontsize=8)
    axd.set_ylim(-0.6, max(n - 0.4, 0.6))
    if n and max(f[1] for f in found) / min(f[1] for f in found) > 3:
        axd.set_xscale("log")
        axd.xaxis.set_major_formatter(ScalarFormatter())
        axd.xaxis.set_minor_formatter(NullFormatter())
    axd.margins(x=0.2)
    axd.set_xlabel("last-10-epoch mean")
    axd.set_title("final level")
    axd.grid(alpha=0.3, axis="x")


def group_figure(name, title, panels, runs_dir, out_dir, rows):
    print(name)
    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 4.4 * len(panels)),
                             gridspec_kw={"width_ratios": [2, 1]}, squeeze=False)
    for r, (ptitle, items) in enumerate(panels):
        panel(axes[r][0], axes[r][1], ptitle, items, runs_dir, name, rows)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.text(0.01, 0.005, FOOT, fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def seed_figure(runs_dir, out_dir, rows):
    name = "8_combinations_and_seeds"
    print(name)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, (cfg, runs) in enumerate(SEEDS.items()):
        vals = []
        for run in runs:
            d = load(runs_dir, run)
            if d is None:
                print(f"  missing: {run}  ({cfg})")
                continue
            m, s = last10(d)
            vals.append(m)
            rows.append(dict(group=name, panel="seeds", label=cfg, run=run, epochs=len(d),
                             last10_mean=m, last10_std=s, best_epoch_value=d.mae.min()))
        if not vals:
            continue
        ax.scatter(i + np.linspace(-0.1, 0.1, len(vals)), vals, s=45, zorder=3, color=f"C{i}")
        ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=f"C{i}", lw=2)
        ax.annotate(f"mean {np.mean(vals):.4f}\nspread {max(vals) - min(vals):.4f}",
                    (i, max(vals)), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels(list(SEEDS), fontsize=9)
    ax.set_ylabel("last-10-epoch validation MAE")
    ax.set_title("Final candidates and seed-to-seed noise (one dot per seed)")
    ax.grid(alpha=0.3, axis="y")
    ax.margins(y=0.2)
    fig.text(0.01, 0.005, "If the spread within a configuration is as large as the gap between "
                         "configurations, the difference is noise.", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="runs/plots")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (title, panels) in GROUPS.items():
        group_figure(name, title, panels, a.runs_dir, out, rows)
    seed_figure(a.runs_dir, out, rows)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "summary.csv", index=False)
        print(f"\n{len(set(df.run))} runs plotted. PNGs and summary.csv are in {out}/")
    else:
        print("no runs found")


if __name__ == "__main__":
    main()
