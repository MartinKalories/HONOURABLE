from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path("/home/manav/PL-NN-testdata_forDec2025/")
DEFAULT_CSV = "bayesopt_20260418-1200_all_trials.csv"
DEFAULT_EXCLUDE = {"trial", "objective_val_loss", "final_val_loss", "run_date", "run_time"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a corner-style plot from a Bayesian optimisation trials CSV."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=(
            "CSV filename or full path. If only a filename is given, it is looked for inside "
            f"{DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output PNG filename or full path. If omitted, saves as <csv_stem>_corner.png "
            f"inside {DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        default=None,
        help="Columns to include in the corner plot. If omitted, varying numeric columns are auto-detected.",
    )
    parser.add_argument(
        "--target",
        default="objective_val_loss",
        help="Column used to colour the scatter points. Default: objective_val_loss",
    )
    parser.add_argument(
        "--log-cols",
        nargs="*",
        default=["learningRate"],
        help="Columns to convert to log10 before plotting. Default: learningRate",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="If > 0, highlight the best N trials based on the target column.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins for diagonal panels.",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=2.8,
        help="Figure size scale per variable. Default: 2.8",
    )
    return parser.parse_args()


def resolve_input_csv(csv_arg: str) -> Path:
    csv_path = Path(csv_arg)
    if csv_path.is_absolute():
        return csv_path
    return DATA_DIR / csv_path



def resolve_output_png(out_arg: str | None, csv_path: Path) -> Path:
    if out_arg is None:
        return DATA_DIR / f"{csv_path.stem}_corner.png"

    out_path = Path(out_arg)
    if out_path.is_absolute():
        return out_path
    return DATA_DIR / out_path



def detect_variable_columns(df: pd.DataFrame, target: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    variable_cols: List[str] = []

    for col in numeric_cols:
        if col in DEFAULT_EXCLUDE or col == target:
            continue
        if df[col].nunique(dropna=False) > 1:
            variable_cols.append(col)

    return variable_cols



def safe_log10(series: pd.Series) -> pd.Series:
    if (series <= 0).any():
        raise ValueError(
            f"Column '{series.name}' contains non-positive values, so log10 cannot be applied."
        )
    return np.log10(series)



def prettify_label(col: str, logged: bool) -> str:
    return f"log10({col})" if logged else col



def make_corner_plot(
    df: pd.DataFrame,
    cols: List[str],
    target: str,
    log_cols: List[str],
    out_path: Path,
    bins: int,
    figscale: float,
    top_n: int,
) -> None:
    if not cols:
        raise ValueError("No columns available to plot.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the CSV.")

    plot_df = df.copy()
    logged_flags: dict[str, bool] = {}

    for col in cols:
        if col not in plot_df.columns:
            raise ValueError(f"Requested column '{col}' was not found in the CSV.")
        if col in log_cols:
            plot_df[col] = safe_log10(plot_df[col])
            logged_flags[col] = True
        else:
            logged_flags[col] = False

    all_needed_cols = list(dict.fromkeys(cols + [target]))
    plot_df = plot_df[all_needed_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if plot_df.empty:
        raise ValueError("No finite rows remain after filtering NaN/inf values.")

    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(figscale * n, figscale * n), squeeze=False)

    cvals = plot_df[target].to_numpy(dtype=float)
    scatter_artist = None

    for i, ycol in enumerate(cols):
        for j, xcol in enumerate(cols):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            x = plot_df[xcol].to_numpy(dtype=float)

            if i == j:
                ax.hist(x, bins=bins)
                ax.set_ylabel("count")
            else:
                y = plot_df[ycol].to_numpy(dtype=float)
                scatter_artist = ax.scatter(x, y, c=cvals, s=24, alpha=0.8)

                if top_n > 0:
                    best_idx = np.argsort(cvals)[:top_n]
                    ax.scatter(x[best_idx], y[best_idx], marker="x", s=80, linewidths=1.5)

            if i == n - 1:
                ax.set_xlabel(prettify_label(xcol, logged_flags[xcol]))
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(prettify_label(ycol, logged_flags[ycol]))
            elif i != j:
                ax.set_yticklabels([])

    if scatter_artist is not None:
        cbar = fig.colorbar(scatter_artist, ax=axes, shrink=0.82, pad=0.02)
        cbar.set_label(target)

    fig.suptitle("Corner plot of varying optimisation variables", y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    args = parse_args()

    csv_path = resolve_input_csv(args.csv)
    out_path = resolve_output_png(args.out, csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}\n"
            f"Put the file in {DATA_DIR} or pass a full path with --csv"
        )

    df = pd.read_csv(csv_path)

    if args.cols is None:
        cols = detect_variable_columns(df, args.target)
    else:
        cols = args.cols

    print(f"Loaded CSV: {csv_path}")
    print(f"Rows, columns: {df.shape}")
    print(f"Using columns: {cols}")
    print(f"Colour target: {args.target}")
    print(f"Saving PNG to: {out_path}")

    make_corner_plot(
        df=df,
        cols=cols,
        target=args.target,
        log_cols=args.log_cols,
        out_path=out_path,
        bins=args.bins,
        figscale=args.figscale,
        top_n=args.top_n,
    )

    print("Done.")


if __name__ == "__main__":
    main()
