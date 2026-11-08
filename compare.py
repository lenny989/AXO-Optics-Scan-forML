"""Compare all runs in runs/: shows only the settings that differ, plus the results.

    python compare.py                 # table, sorted by best validation error
    python compare.py --plot          # also saves runs/comparison.png (validation curves)
"""
import argparse
import json
from pathlib import Path

import pandas as pd

IGNORE = {"name", "resume", "workers", "runs_dir", "max_batches"}   # don't affect the result
RESULT_COLS = ["best_val_mean", "best_val_dx", "best_val_dz", "best_epoch",
               "last_epoch", "baseline_mean", "hours"]


def load_runs(runs_dir):
    rows = []
    for d in sorted(Path(runs_dir).iterdir()):
        if not ((d / "config.json").exists() and (d / "summary.json").exists()):
            continue
        row = {"run": d.name}
        row |= json.loads((d / "config.json").read_text())
        row |= json.loads((d / "summary.json").read_text())
        log = d / "log.csv"
        row["hours"] = round(pd.read_csv(log).seconds.sum() / 3600, 2) if log.exists() else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()

    df = load_runs(a.runs_dir)
    if df.empty:
        raise SystemExit(f"no finished runs found in {a.runs_dir}/")
    df = df.sort_values("best_val_mean")

    # Settings that differ between runs = what you are actually comparing.
    cfg_cols = [c for c in df.columns if c not in IGNORE | set(RESULT_COLS) | {"run"}
                and c not in ("last_train_loss", "n_params", "baseline_dx", "baseline_dz")]
    varying = [c for c in cfg_cols if df[c].astype(str).nunique() > 1]

    if "val_scans" in varying:
        print("WARNING: runs use different validation scans, so their errors are not comparable.\n")

    show = ["run"] + varying + [c for c in RESULT_COLS if c in df.columns]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df[show].round(4).to_string(index=False))

    df.to_csv(Path(a.runs_dir) / "comparison.csv", index=False)
    print(f"\nfull table (all settings) saved to {a.runs_dir}/comparison.csv")

    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for run in df.run:
            log = pd.read_csv(Path(a.runs_dir) / run / "log.csv")
            ax.plot(log.epoch, (log.val_mae_dx + log.val_mae_dz) / 2, label=run)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation MAE (mean of dx, dz)")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(Path(a.runs_dir) / "comparison.png", dpi=150)
        print(f"plot saved to {a.runs_dir}/comparison.png")


if __name__ == "__main__":
    main()