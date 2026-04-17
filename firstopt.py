from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DATA_DIR = Path("/home/manav/PL-NN-testdata_forDec2025/")
DEFAULT_CSV = "bayesopt_20260416-1046_all_trials.csv"
DEFAULT_EXCLUDE = {
    "trial",
    "objective_val_loss",
    "final_val_loss",
    "run_date",
    "run_time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a smoother corner plot from a Bayesian optimisation trials CSV."
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
            "Output PNG filename or full path. If omitted, saves as <csv_stem>_corner_kde.png "
            f"inside {DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--cols",
        nargs="+",
        default=None,
        help="Columns to include. If omitted, varying numeric columns are auto-detected.",
    )
    parser.add_argument(
        "--target",
        default="objective_val_loss",
        help="Loss column used to rank and highlight best trials.",
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
        default=3,
        help="Highlight the best N trials. Default: 3",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=18,
        help="Histogram bins on the diagonal. Default: 18",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=2.7,
        help="Figure size scale per variable. Default: 2.7",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=6,
        help="Number of KDE contour levels. Default: 6",
    )
    parser.add_argument(
        "--jitter-frac",
        type=float,
        default=0.015,
        help="Small fractional jitter added to discrete columns for KDE stability. Default: 0.015",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible jitter. Default: 42",
    )
    parser.add_argument(
        "--no-kde",
        action="store_true",
        help="Disable KDE contours and just show scatter + histograms.",
    )
    return parser.parse_args()


def resolve_input_csv(csv_arg: str) -> Path:
    csv_path = Path(csv_arg)
    return csv_path if csv_path.is_absolute() else DATA_DIR / csv_path


def resolve_output_png(out_arg: str | None, csv_path: Path) -> Path:
    if out_arg is None:
        return DATA_DIR / f"{csv_path.stem}_corner_kde.png"
    out_path = Path(out_arg)
    return out_path if out_path.is_absolute() else DATA_DIR / out_path


def detect_variable_columns(df: pd.DataFrame, target: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols: list[str] = []
    for col in numeric_cols:
        if col in DEFAULT_EXCLUDE or col == target:
            continue
        if df[col].nunique(dropna=False) > 1:
            cols.append(col)
    return cols


def safe_log10(series: pd.Series) -> pd.Series:
    if (series <= 0).any():
        raise ValueError(f"Column '{series.name}' contains non-positive values, so log10 cannot be applied.")
    return np.log10(series)


def prettify_label(col: str, logged: bool) -> str:
    return f"log10({col})" if logged else col


def is_integer_like(values: np.ndarray) -> bool:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return False
    return np.allclose(vals, np.round(vals), atol=1e-10)


def maybe_add_jitter(values: np.ndarray, jitter_frac: float, rng: np.random.Generator) -> np.ndarray:
    vals = np.asarray(values, dtype=float).copy()
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return vals

    uniq = np.unique(finite)
    if is_integer_like(finite) or uniq.size <= max(10, int(0.35 * finite.size)):
        spread = np.ptp(finite)
        if spread == 0:
            spread = max(abs(finite[0]), 1.0)
        noise = rng.normal(loc=0.0, scale=jitter_frac * spread, size=vals.shape)
        vals = vals + noise
    return vals


def kde_1d_line(ax: plt.Axes, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5 or np.unique(x).size < 3:
        return
    xmin, xmax = np.min(x), np.max(x)
    pad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    grid = np.linspace(xmin - pad, xmax + pad, 250)
    kde = gaussian_kde(x)
    dens = kde(grid)
    ax2 = ax.twinx()
    ax2.plot(grid, dens, linewidth=1.2)
    ax2.set_yticks([])
    ax2.set_ylabel("")


def kde_2d_contours(ax: plt.Axes, x: np.ndarray, y: np.ndarray, levels: int) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 8:
        return
    if np.unique(x).size < 4 or np.unique(y).size < 4:
        return

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xpad = 0.08 * (xmax - xmin if xmax > xmin else 1.0)
    ypad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)

    xx, yy = np.meshgrid(
        np.linspace(xmin - xpad, xmax + xpad, 140),
        np.linspace(ymin - ypad, ymax + ypad, 140),
    )
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zmin = np.min(zz)
    zmax = np.max(zz)
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return

    contour_levels = np.linspace(zmin + 0.18 * (zmax - zmin), zmax, levels)
    ax.contourf(xx, yy, zz, levels=contour_levels, alpha=0.45)
    ax.contour(xx, yy, zz, levels=contour_levels, linewidths=0.8)



def try_corner_package(data: np.ndarray, labels: list[str], out_path: Path) -> bool:
    try:
        import corner  # type: ignore
    except Exception:
        return False

    fig = corner.corner(
        data,
        labels=labels,
        show_titles=False,
        plot_contours=True,
        fill_contours=True,
        bins=18,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True



def make_corner_plot(
    df: pd.DataFrame,
    cols: list[str],
    target: str,
    log_cols: Iterable[str],
    out_path: Path,
    bins: int,
    figscale: float,
    top_n: int,
    levels: int,
    jitter_frac: float,
    seed: int,
    use_kde: bool,
) -> None:
    if not cols:
        raise ValueError("No columns available to plot.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the CSV.")

    plot_df = df.copy()
    logged_flags: dict[str, bool] = {}
    log_cols = set(log_cols)

    for col in cols:
        if col not in plot_df.columns:
            raise ValueError(f"Requested column '{col}' was not found in the CSV.")
        if col in log_cols:
            plot_df[col] = safe_log10(plot_df[col])
            logged_flags[col] = True
        else:
            logged_flags[col] = False

    needed = list(dict.fromkeys(cols + [target]))
    plot_df = plot_df[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if plot_df.empty:
        raise ValueError("No finite rows remain after filtering NaN/inf values.")

    labels = [prettify_label(col, logged_flags[col]) for col in cols]
    data_matrix = plot_df[cols].to_numpy(dtype=float)

    # If the external corner package is available, use it.
    # It produces the most familiar 'corner plot' look.
    if use_kde and try_corner_package(data_matrix, labels, out_path):
        return

    rng = np.random.default_rng(seed)
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(figscale * n, figscale * n), squeeze=False)

    cvals = plot_df[target].to_numpy(dtype=float)
    best_idx = np.argsort(cvals)[: max(top_n, 0)] if top_n > 0 else np.array([], dtype=int)

    # Precompute jittered columns only for KDE use
    jittered: dict[str, np.ndarray] = {}
    for col in cols:
        jittered[col] = maybe_add_jitter(plot_df[col].to_numpy(dtype=float), jitter_frac, rng)

    for i, ycol in enumerate(cols):
        for j, xcol in enumerate(cols):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            x = plot_df[xcol].to_numpy(dtype=float)

            if i == j:
                ax.hist(x, bins=bins, density=True, alpha=0.8)
                kde_1d_line(ax, jittered[xcol])
                ax.set_ylabel("density")
            else:
                y = plot_df[ycol].to_numpy(dtype=float)
                if use_kde:
                    kde_2d_contours(ax, jittered[xcol], jittered[ycol], levels=levels)
                ax.scatter(x, y, s=14, alpha=0.75)
                if best_idx.size:
                    ax.scatter(
                        x[best_idx],
                        y[best_idx],
                        marker="x",
                        s=70,
                        linewidths=1.4,
                    )

            if i == n - 1:
                ax.set_xlabel(prettify_label(xcol, logged_flags[xcol]))
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(prettify_label(ycol, logged_flags[ycol]))
            elif i != j:
                ax.set_yticklabels([])

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
    cols = detect_variable_columns(df, args.target) if args.cols is None else args.cols

    print(f"Loaded CSV: {csv_path}")
    print(f"Rows, columns: {df.shape}")
    print(f"Using columns: {cols}")
    print(f"Target column: {args.target}")
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
        levels=args.levels,
        jitter_frac=args.jitter_frac,
        seed=args.seed,
        use_kde=not args.no_kde,
    )

    print("Done.")


if __name__ == "__main__":
    main()
