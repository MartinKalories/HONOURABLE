from pathlib import Path
from itertools import combinations

# import matplotlib
# matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.optimize import differential_evolution, brentq
from skopt.space import Real, Integer, Categorical
from optim_space import space
from datetime import datetime
# ==================================================
# File path
# ==================================================
DATA_DIR = Path("/home/manav/PL-NN-testdata_forDec2025/")
#DATA_DIR = Path("./data/")
DEFAULT_CSV = "bayesopt_current_nosub_noLR_all_trials)_w_50extra_all_trials.csv"

csv_path = DATA_DIR / DEFAULT_CSV
df = pd.read_csv(csv_path)

print("Loaded:", csv_path)


# ==================================================
# USER SETTINGS
# Every parameter that changes the analysis results is set here.
# ==================================================

# Column in the CSV used as the loss (lower = better).
loss_col = "objective_val_loss"

# Temperature for the trial weights: w = exp(-(loss - min_loss) / T).
# Same units as the loss.
# Lower T makes only the best trials matter strongly.
# Higher T gives smoother weighting across more trials.
# If T is much larger than the spread of the losses, every trial gets almost the
# same weight and the KDEs only show where the optimiser placed trials.
# None = choose T automatically so the effective number of trials is TARGET_N_EFF
# (recommended, because a sensible T depends on how spread out the losses are).
# A number = use that T directly.
T = None

# Effective number of trials, n_eff = 1 / sum(w^2), used to choose T when T is None.
# Smaller = only the very best trials count (more selective, but noisier KDEs).
# Larger = more trials count (smoother, but less selective).
TARGET_N_EFF = 30

# Bandwidth for the 1D KDEs and the joint continuous KDE (scipy gaussian_kde).
# None or "scott" = Scott's rule, "silverman" = Silverman's rule,
# a number = scale factor applied to the weighted covariance.
BW_METHOD = None

# Scale factor for the Scott's-rule bandwidths in the 2D KDEs:
# bw = BW_SCALE_2D * std * n_eff^(-1/6), where n_eff is the effective number of trials.
# Below 1 gives sharper, more detailed contours; above 1 gives smoother ones.
# Does not affect variables listed in MANUAL_BW.
BW_SCALE_2D = 1.0

# Fixed bandwidths for the 2D KDEs, in the variable's own units.
# Variables not listed here use Scott's rule scaled by BW_SCALE_2D.
MANUAL_BW = {
    "ksz_psf": 1,
    "ksz_wf": 1,

    "nfilts_psf": 32,
    "nfilts_wf": 32,
    "nfilts_enc": 32,
}

# Edge correction (reflection method). A Gaussian kernel near the edge of the search
# space puts part of its mass outside the range, so without correction the KDE is
# artificially low near the edges (about half height right at an edge).
# True = add mirror-image kernels about each bound, folding that mass back inside.
# Bounds are the search-space bounds from optim_space.py (for discrete variables,
# the smallest and largest values tried).
# Applies to the 1D curves, the 2D panels and the joint continuous optimum.
# False = no edge correction.
REFLECT_AT_BOUNDS = False

# What the 1D curves and 2D panels show.
# False = weighted KDE (previous behaviour): density of trials, each weighted by its
#         Boltzmann weight. High where many trials were run, as well as where trials
#         were good.
# True  = kernel regression: weighted KDE / unweighted KDE, with the same bandwidths.
#         This is the local average Boltzmann weight around each point, normalised to
#         unit area. It does not depend on how many trials were run in each region.
# The joint continuous optimum always uses the weighted KDE.
USE_KERNEL_REGRESSION = True

# Settings for differential_evolution when finding the joint continuous KDE optimum.
DE_SEED = 42
DE_POLISH = True


# ==================================================
# OUTPUT SETTINGS
# Choose which plots and summaries to produce (True = make it).
# ==================================================
MAKE_CORNER_PLOT = True              # 2D_KDE_corner_plot_*.png
SHOW_CORNER_PLOT = True              # keep the corner plot window open after the script finishes
SQUARE_CORNER_PANELS = False          # force every corner plot panel to be square (False = matplotlib's default layout)
MAKE_1D_KDE_PLOTS = False            # 1D_*.png and KDE_1D_*.csv
MAKE_2D_KDE_PLOTS = False            # 2D_KDE_*_vs_*.png and KDE_2D_ALL_VARIABLES_*.csv
MAKE_KERNEL_GROUPED_1D_PLOTS = False # 1D_KDE_*_grouped_by_ksz_*.png
MAKE_DISCRETE_EFFECT_PLOTS = False   # variable_effect_*.png and DISCRETE_VARIABLE_EFFECTS_*.csv
MAKE_CONTINUOUS_EFFECT_PLOTS = False # continuous_effect_*.png
SAVE_BEST_TRIAL_SUMMARY = False      # BEST_TRIAL_*.csv


# ==================================================
# Variables from the optimiser search space (optim_space.py)
# ==================================================
active_dims = [dim for dim in space if dim.name is not None]

continuous_params = [
    dim.name
    for dim in active_dims
    if isinstance(dim, (Real, Integer))
]

discrete_params = [
    dim.name
    for dim in active_dims
    if isinstance(dim, Categorical)
]

all_params = [
    dim.name
    for dim in active_dims
]

continuous_labels = continuous_params.copy()

space_by_name = {
    dim.name: dim
    for dim in active_dims
}


# ==================================================
# Basic checks
# ==================================================
required_cols = ["trial", loss_col] + all_params
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(
        "The CSV is missing these columns:\n"
        + "\n".join(missing_cols)
    )

df = df.copy()

# Remove rows with missing or non-finite loss
df = df[np.isfinite(df[loss_col])].copy()

if len(df) == 0:
    raise ValueError("No valid rows left after removing invalid losses.")

loss = df[loss_col].to_numpy(dtype=float)


# ==================================================
# Output folder
# ==================================================
# Timestamp for this run, for example: 20260706_183542
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = (
    csv_path.parent
    / f"KDE_{csv_path.stem}_plots_fixed_bw_{timestamp}"
)

output_dir.mkdir(parents=True, exist_ok=True)
print("Plots will be saved to:", output_dir)


# ==================================================
# Convert loss into goodness weights
# Lower loss = better
# ==================================================
def boltzmann_weights(loss, T):
    goodness = np.exp(-(loss - loss.min()) / T)
    return goodness / goodness.sum()


def effective_sample_size(w):
    """Kish effective number of trials, 1 / sum(w^2), for weights that sum to 1."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return 1.0 / np.sum(w ** 2)


def temperature_for_n_eff(loss, target_n_eff):
    """Finds the T at which the Boltzmann weights have the requested n_eff."""
    n = len(loss)
    spread = loss.max() - loss.min()

    if not 1 < target_n_eff < n:
        raise ValueError(f"TARGET_N_EFF must be between 1 and the number of trials ({n}).")

    if spread <= 0:
        raise ValueError("All losses are equal, so T cannot be chosen from n_eff.")

    # n_eff rises from ~1 (T -> 0) to n (T -> infinity), so solve for log(T)
    def n_eff_error(log_T):
        return effective_sample_size(boltzmann_weights(loss, np.exp(log_T))) - target_n_eff

    log_T = brentq(n_eff_error, np.log(spread * 1e-6), np.log(spread * 1e6))

    return float(np.exp(log_T))


if T is None:
    T = temperature_for_n_eff(loss, TARGET_N_EFF)
    print(f"T chosen automatically for n_eff = {TARGET_N_EFF}")

weights = boltzmann_weights(loss, T)

print(
    f"Trial weights: T = {T:.4g} (loss range {loss.max() - loss.min():.4g}), "
    f"effective number of trials = {effective_sample_size(weights):.1f} of {len(weights)}"
)

df["kde_weight"] = weights


# ==================================================
# Continuous samples for KDE
# ==================================================
def make_continuous_samples(df):
    return df[continuous_params].astype(float).to_numpy(dtype=float)


samples = make_continuous_samples(df)


# ==================================================
# KDE helper functions
# ==================================================
def kde_bandwidth(data, weights=None, bw_method=BW_METHOD):
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    try:
        kde = gaussian_kde(
            data,
            weights=weights,
            bw_method=bw_method,
        )

        return np.sqrt(np.diag(kde.covariance))

    except np.linalg.LinAlgError:
        return None

def variable_bounds(var_name, values):
    """
    Range used for the KDE grid and the edge reflection.
    Continuous variables: the search-space bounds from optim_space.py.
    Anything else: the smallest and largest values in the data.
    """
    dim = space_by_name.get(var_name)

    if isinstance(dim, (Real, Integer)):
        return float(dim.low), float(dim.high)

    values = np.asarray(values, dtype=float)
    return float(values.min()), float(values.max())


def axis_kernels(grid, data, bw, bounds):
    """
    Gaussian kernel values along one axis, shape (n_data, n_grid).
    With REFLECT_AT_BOUNDS, each trial also gets mirror-image kernels about
    both bounds, so kernel mass that would spill past an edge is folded back.
    """
    centres = [data]

    if REFLECT_AT_BOUNDS:
        lo, hi = bounds
        centres += [2 * lo - data, 2 * hi - data]

    k = sum(
        np.exp(-0.5 * ((grid[np.newaxis, :] - c[:, np.newaxis]) / bw) ** 2)
        for c in centres
    )

    return k / (np.sqrt(2 * np.pi) * bw)


def combine_kernel_sums(weighted_sum, unweighted_sum):
    """
    Weighted KDE, or with USE_KERNEL_REGRESSION the kernel regression
    weighted KDE / unweighted KDE (the local average weight).
    """
    if not USE_KERNEL_REGRESSION:
        return weighted_sum

    return weighted_sum / np.maximum(unweighted_sum, 1e-300)


def kde_1d(x, w, grids=400, bw_method=BW_METHOD, var_name=None):
    """
    1D weighted KDE (or kernel regression, see USE_KERNEL_REGRESSION) on a
    grid spanning variable_bounds(var_name), normalised to unit area.
    The bandwidth comes from scipy's gaussian_kde (BW_METHOD).
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()

    if np.std(x) < 1e-12:
        return None, None

    bw = kde_bandwidth(x, w, bw_method=bw_method)

    if bw is None:
        return None, None

    bounds = variable_bounds(var_name, x)
    grid = np.linspace(bounds[0], bounds[1], grids)

    K = axis_kernels(grid, x, bw[0], bounds)
    pdf = combine_kernel_sums(w @ K, K.mean(axis=0))

    area = trapezoid(pdf, grid)
    if area > 0:
        pdf /= area

    return grid, pdf


def get_bandwidth(values, var_name, d=1, weights=None):
    """
    Returns the KDE bandwidth for one variable.

    If var_name is in MANUAL_BW, it uses the manual value.
    Otherwise it uses Scott's-rule style bandwidth, multiplied by BW_SCALE_2D,
    with the effective number of trials from the weights (all trials if no weights).
    """

    values = np.asarray(values, dtype=float)

    if var_name in MANUAL_BW:
        return MANUAL_BW[var_name]

    std = np.std(values, ddof=1)
    n = len(values) if weights is None else effective_sample_size(weights)

    if std < 1e-12 or n <= 1:
        return None

    return BW_SCALE_2D * std * n ** (-1 / (d + 4))

def kde_2d(x, y, w, x_name=None, y_name=None, grids=200):
    """
    2D weighted KDE (or kernel regression, see USE_KERNEL_REGRESSION) with
    separate bandwidths for x and y, on a grid spanning variable_bounds.

    If x_name or y_name appears in MANUAL_BW, that manual bandwidth is used.
    Otherwise Scott's-rule style bandwidth is used.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    # Normalise weights
    w = w / (np.sum(w) + 1e-12)

    x_bounds = variable_bounds(x_name, x)
    y_bounds = variable_bounds(y_name, y)

    xg = np.linspace(x_bounds[0], x_bounds[1], grids)
    yg = np.linspace(y_bounds[0], y_bounds[1], grids)

    X, Y = np.meshgrid(xg, yg)

    bw_x = get_bandwidth(x, x_name, d=2, weights=w)
    bw_y = get_bandwidth(y, y_name, d=2, weights=w)

    if bw_x is None or bw_y is None:
        return X, Y, None

    Kx = axis_kernels(xg, x, bw_x, x_bounds)
    Ky = axis_kernels(yg, y, bw_y, y_bounds)

    # Sum over trials of w_i * Ky_i(y) * Kx_i(x), with rows = y to match meshgrid
    weighted_sum = (Ky * w[:, np.newaxis]).T @ Kx
    unweighted_sum = Ky.T @ Kx / len(x)

    Z = combine_kernel_sums(weighted_sum, unweighted_sum)

    if USE_KERNEL_REGRESSION:
        Z /= trapezoid(trapezoid(Z, xg, axis=1), yg)

    return X, Y, Z


# ==================================================
# Find continuous 5D KDE optimum
# ==================================================
def find_5d_kde_optimum(df, loss, csv_path, T, bw_method=BW_METHOD):
    samples_5d = make_continuous_samples(df)

    weights_5d = boltzmann_weights(loss, T)

    try:
        kde = gaussian_kde(
            samples_5d.T,
            weights=weights_5d,
            bw_method=bw_method,
        )
    except np.linalg.LinAlgError:
        print("Could not compute 5D KDE optimum because the covariance matrix was singular.")
        return None, None

    # Search over the full search space, not just the range of the trials
    bounds = [
        variable_bounds(name, samples_5d[:, k])
        for k, name in enumerate(continuous_params)
    ]

    def negative_kde_density(x):
        points = np.asarray(x, dtype=float).reshape(len(continuous_params), 1)

        # Edge correction: evaluating at the point and at its mirror images about
        # every bound equals adding mirror-image kernels (exact even though the
        # kernel covariance has correlations between variables)
        if REFLECT_AT_BOUNDS:
            for k, (lo, hi) in enumerate(bounds):
                low_mirror = points.copy()
                low_mirror[k] = 2 * lo - low_mirror[k]
                high_mirror = points.copy()
                high_mirror[k] = 2 * hi - high_mirror[k]
                points = np.concatenate([points, low_mirror, high_mirror], axis=1)

        return -kde(points).sum()

    result = differential_evolution(
        negative_kde_density,
        bounds=bounds,
        seed=DE_SEED,
        polish=DE_POLISH,
    )

    optimum_5d_transformed = result.x
    peak_density_5d = -result.fun

    optimum_physical = {}

    for i, name in enumerate(continuous_params):
        value = float(optimum_5d_transformed[i])
        dim = space_by_name.get(name)

    # If the optimiser variable is an Integer, save it as an integer
        if isinstance(dim, Integer):
            optimum_physical[name] = int(round(value))
        else:
            optimum_physical[name] = value

    optimum_physical["kde_peak_density"] = float(peak_density_5d)
    optimum_physical["kde_dimension"] = len(continuous_params)
    optimum_physical["T"] = T

    output_path = csv_path.parent / f"KDE_5D_{csv_path.name}"
    pd.DataFrame([optimum_physical]).to_csv(output_path, index=False)

    print("\n5D continuous KDE optimum:")
    for k, v in optimum_physical.items():
        print(f"{k}: {v}")

    print("5D KDE result saved to:", output_path)

    return optimum_5d_transformed, optimum_physical


def make_5d_point_for_2d_plots(optimum_5d_transformed):
    if optimum_5d_transformed is None:
        return None

    point_5d = np.full(4, np.nan)

    point_5d[0] = optimum_5d_transformed[0]  # dropout_rate
    point_5d[1] = optimum_5d_transformed[1]  # dropout_rate_dense
    point_5d[2] = optimum_5d_transformed[2]  # dropout_rate_psf
    point_5d[3] = optimum_5d_transformed[3]  # n_units_dense

    return point_5d


# ==================================================
# Plot 1D continuous KDE graphs
# ==================================================
def plot_kde_1d(samples, weights, labels, csv_path, grids=400):
    results = []

    for k in range(samples.shape[1]):
        x = samples[:, k]
        bw = kde_bandwidth(x, weights)

        if bw is not None:
            bw_label = f"{bw[0]:.3g}"
            print(f"1D KDE bandwidth for {labels[k]}: {bw_label}")
        else:
            bw_label = "NA"

        grid, pdf = kde_1d(x, weights, grids=grids, var_name=labels[k])

        if grid is None:
            print(f"Skipping 1D KDE for {labels[k]}: not enough variation.")
            continue

        max_idx = np.argmax(pdf)
        kde_peak = grid[max_idx]
        peak_density = pdf[max_idx]

        plt.figure(figsize=(6, 4))
        plt.plot(grid, pdf)

        plt.scatter(
            kde_peak,
            peak_density,
            s=80,
            color="black",
            zorder=10,
            label="1D KDE peak",
        )

        plt.xlabel(labels[k])
        plt.ylabel("Marginal PDF")
        plt.title(f"1D KDE: {labels[k]}|  bandwidth = {bw_label}")
        plt.legend()
        plt.tight_layout()

        filename = (
            f"1D_{labels[k]}.png"
            .replace("(", "")
            .replace(")", "")
            .replace("/", "")
        )

        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()

        results.append({
            "param": labels[k],
            "kde_1d_peak": kde_peak,
            "peak_density": peak_density,
        })

    results_df = pd.DataFrame(results)
    output_path = csv_path.parent / f"KDE_1D_{csv_path.name}"
    results_df.to_csv(output_path, index=False)

    print("1D KDE plots saved to:", output_dir)
    print("1D KDE results saved to:", output_path)


# ==================================================
# Plot 2D continuous KDE pair graphs
# ==================================================
def plot_kde_pairs(samples, weights, labels, csv_path, kde5d_point=None, grids=200, levels=30):
    for i, j in combinations(range(samples.shape[1]), 2):
        X, Y, Z = kde_2d(
            samples[:, i],
            samples[:, j],
            weights,
            grids=grids,
        )

        if Z is None:
            print(f"Skipping 2D KDE for {labels[i]} vs {labels[j]}: not enough variation.")
            continue

        plt.figure(figsize=(6, 5))

        cf = plt.contourf(
            X,
            Y,
            Z,
            levels=levels,
            cmap="viridis",
        )

        plt.contour(
            X,
            Y,
            Z,
            levels=levels,
            colors="k",
            alpha=0.35,
            linewidths=0.5,
        )

        plt.scatter(
            samples[:, i],
            samples[:, j],
            s=25,
            alpha=0.45,
            label="Trials",
        )

        # 2D KDE peak
        max_idx = np.unravel_index(np.argmax(Z), Z.shape)
        kde_peak_x = X[max_idx]
        kde_peak_y = Y[max_idx]

        plt.scatter(
            kde_peak_x,
            kde_peak_y,
            s=120,
            marker="o",
            edgecolor="white",
            linewidths=2,
            color="black",
            label="2D KDE peak",
            zorder=10,
        )

        # 5D KDE optimum projected onto this 2D plane
        if kde5d_point is not None:
            x5 = kde5d_point[i]
            y5 = kde5d_point[j]

            if not np.isnan(x5) and not np.isnan(y5):
                plt.scatter(
                    x5,
                    y5,
                    s=150,
                    marker="x",
                    color="red",
                    linewidths=3,
                    label="5D KDE optimum",
                    zorder=11,
                )

        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.title(f"KDE: {labels[i]} vs {labels[j]}")
        plt.colorbar(cf, label="Weighted KDE density")
        plt.legend()
        plt.tight_layout()

        filename = (
            f"{labels[i]}_vs_{labels[j]}.png"
            .replace("(", "")
            .replace(")", "")
            .replace("/", "")
        )

        save_path = output_dir / filename
        plt.savefig(save_path, dpi=300)
        plt.close()

    print("2D KDE plots saved to:", output_dir)


# ==================================================
# Save 2D KDE peak numerical results
# ==================================================
def save_kde_results(samples, weights, labels, csv_path, kde5d_point=None, grids=200):
    results = []

    for i, j in combinations(range(samples.shape[1]), 2):
        X, Y, Z = kde_2d(
            samples[:, i],
            samples[:, j],
            weights,
            grids=grids,
        )

        if Z is None:
            results.append({
                "param_x": labels[i],
                "param_y": labels[j],
                "kde_peak_x": np.nan,
                "kde_peak_y": np.nan,
                "peak_density": np.nan,
                "kde5d_x": np.nan,
                "kde5d_y": np.nan,
                "status": "skipped_not_enough_variation",
            })
            continue

        max_idx = np.unravel_index(np.argmax(Z), Z.shape)

        row = {
            "param_x": labels[i],
            "param_y": labels[j],
            "kde_peak_x": X[max_idx],
            "kde_peak_y": Y[max_idx],
            "peak_density": Z[max_idx],
            "status": "ok",
        }

        if kde5d_point is not None:
            row["kde5d_x"] = kde5d_point[i]
            row["kde5d_y"] = kde5d_point[j]
        else:
            row["kde5d_x"] = np.nan
            row["kde5d_y"] = np.nan

        results.append(row)

    results_df = pd.DataFrame(results)

    output_path = csv_path.parent / f"KDE_2D_{csv_path.name}"
    results_df.to_csv(output_path, index=False)

    print("2D KDE results saved to:", output_path)


# ==================================================
# Helper for safe filenames
# ==================================================
def safe_filename(text):
    return (
        str(text)
        .replace("(", "")
        .replace(")", "")
        .replace("/", "")
        .replace("\\", "")
        .replace(" ", "_")
        .replace(":", "")
    )


# ==================================================
# Encode discrete variables for 2D KDE
# ==================================================
def encode_discrete_variables(df, discrete_params):
    """
    Converts discrete variables into numeric values so they can be used
    in 2D KDE plots.

    Numeric discrete variables like kernel sizes and filter counts are kept
    as their actual values.

    Non-numeric variables like actFunc are converted into category codes.
    """

    encoded = pd.DataFrame(index=df.index)
    tick_info = {}

    for col in discrete_params:
        numeric_values = pd.to_numeric(df[col], errors="coerce")

        # If the whole column is numeric, keep the actual values
        if numeric_values.notna().all():
            encoded[col] = numeric_values.astype(float)

            unique_vals = np.sort(encoded[col].unique())

            tick_info[col] = {
                "ticks": unique_vals,
                "labels": [
                    str(int(v)) if float(v).is_integer() else f"{v:g}"
                    for v in unique_vals
                ],
            }

        # Otherwise treat it as categorical
        else:
            categories = sorted(df[col].astype(str).unique())
            category_to_code = {cat: i for i, cat in enumerate(categories)}

            encoded[col] = df[col].astype(str).map(category_to_code).astype(float)

            tick_info[col] = {
                "ticks": np.arange(len(categories)),
                "labels": categories,
            }

    return encoded, tick_info


def add_discrete_jitter(values, rng, frac=0.035):
    """
    Adds small jitter to discrete scatter points so repeated trials
    do not sit directly on top of each other.
    """

    values = np.asarray(values, dtype=float)
    unique_vals = np.sort(np.unique(values))

    if len(unique_vals) < 2:
        return np.zeros_like(values)

    min_step = np.min(np.diff(unique_vals))

    return rng.normal(0, frac * min_step, size=len(values))


def nearest_discrete_label(col, value, tick_info):
    """
    Converts a KDE peak coordinate back to the nearest real discrete value.
    """

    ticks = np.asarray(tick_info[col]["ticks"], dtype=float)
    labels = tick_info[col]["labels"]

    nearest_idx = np.argmin(np.abs(ticks - value))

    return labels[nearest_idx]


# ==================================================
# Build all variables for 2D KDE: continuous + discrete
# ==================================================
def make_all_kde_variables(df):
    X = pd.DataFrame(index=df.index)
    variable_info = {}

    # ------------------------------
    # Continuous variables from optimiser space
    # ------------------------------
    for col in continuous_params:
        X[col] = df[col].astype(float)

        variable_info[col] = {
            "label": col,
            "type": "continuous",
        }

    # ------------------------------
    # Discrete variables from optimiser space
    # ------------------------------
    for col in discrete_params:
        numeric_values = pd.to_numeric(df[col], errors="coerce")

        if numeric_values.notna().all():
            X[col] = numeric_values.astype(float)

            unique_vals = np.sort(X[col].unique())

            variable_info[col] = {
                "label": col,
                "type": "discrete_numeric",
                "ticks": unique_vals,
                "ticklabels": [
                    str(int(v)) if float(v).is_integer() else f"{v:g}"
                    for v in unique_vals
                ],
            }

        else:
            categories = sorted(df[col].astype(str).unique())
            category_to_code = {cat: i for i, cat in enumerate(categories)}

            X[col] = df[col].astype(str).map(category_to_code).astype(float)

            variable_info[col] = {
                "label": col,
                "type": "discrete_categorical",
                "ticks": np.arange(len(categories)),
                "ticklabels": categories,
            }

    return X, variable_info


def nearest_axis_value(var_name, value, variable_info):
    """
    For discrete variables, converts a KDE peak coordinate back to the
    nearest real category/value.
    """

    info = variable_info[var_name]

    if info["type"] == "continuous":
        return value

    ticks = np.asarray(info["ticks"], dtype=float)
    ticklabels = info["ticklabels"]

    nearest_idx = np.argmin(np.abs(ticks - value))

    return ticklabels[nearest_idx]


# ==================================================
# 2D KDE plots for all variable pairs
# continuous-continuous, continuous-discrete, discrete-discrete
# ==================================================
def plot_all_2d_kde_pairs(
    df,
    weights,
    csv_path,
    kde5d_continuous_point=None,
    grids=200,
    levels=30,
):
    """
    Makes 2D KDE plots for every pair of variables.

    This replaces:
    - plot_kde_pairs(...)
    - save_kde_results(...)
    - plot_discrete_2d_kde_pairs(...)

    It includes mixed continuous/discrete projections.
    """

    X_all, variable_info = make_all_kde_variables(df)

    columns = list(X_all.columns)
    results = []

    best_idx = df[loss_col].idxmin()

    for col_x, col_y in combinations(columns, 2):
        x = X_all[col_x].to_numpy(dtype=float)
        y = X_all[col_y].to_numpy(dtype=float)

        x_label = variable_info[col_x]["label"]
        y_label = variable_info[col_y]["label"]
        bw_x = get_bandwidth(x, col_x, d=2, weights=weights)
        bw_y = get_bandwidth(y, col_y, d=2, weights=weights)

        if bw_x is not None and bw_y is not None:
            bw_x_label = f"{bw_x:.3g}"
            bw_y_label = f"{bw_y:.3g}"

            print(
                f"2D KDE bandwidth for {x_label} vs {y_label}: "
                f"{x_label} = {bw_x_label}, {y_label} = {bw_y_label}"
            )
        else:
            bw_x_label = "NA"
            bw_y_label = "NA"

        X_grid, Y_grid, Z = kde_2d(
            x,
            y,
            weights,
            x_name=col_x,
            y_name=col_y,
            grids=grids,
        )

        

        if Z is None:
            print(f"Skipping 2D KDE for {x_label} vs {y_label}: not enough variation.")

            results.append({
                "param_x": col_x,
                "param_y": col_y,
                "param_x_label": x_label,
                "param_y_label": y_label,
                "kde_peak_x": np.nan,
                "kde_peak_y": np.nan,
                "kde_peak_x_nearest_value": np.nan,
                "kde_peak_y_nearest_value": np.nan,
                "peak_density": np.nan,
                "status": "skipped_not_enough_variation",
            })

            continue

        # KDE peak
        max_idx = np.unravel_index(np.argmax(Z), Z.shape)

        peak_x = X_grid[max_idx]
        peak_y = Y_grid[max_idx]
        peak_density = Z[max_idx]

        peak_x_nearest = nearest_axis_value(col_x, peak_x, variable_info)
        peak_y_nearest = nearest_axis_value(col_y, peak_y, variable_info)

        results.append({
            "param_x": col_x,
            "param_y": col_y,
            "param_x_label": x_label,
            "param_y_label": y_label,
            "kde_peak_x": peak_x,
            "kde_peak_y": peak_y,
            "kde_peak_x_nearest_value": peak_x_nearest,
            "kde_peak_y_nearest_value": peak_y_nearest,
            "peak_density": peak_density,
            "status": "ok",
        })

        plt.figure(figsize=(7, 5.5))

        cf = plt.contourf(
            X_grid,
            Y_grid,
            Z,
            levels=levels,
            cmap="viridis",
        )

        plt.contour(
            X_grid,
            Y_grid,
            Z,
            levels=levels,
            colors="black",
            alpha=0.35,
            linewidths=0.5,
        )

        # Raw trials coloured by loss
        sc = plt.scatter(
            x,
            y,
            c=df[loss_col],
            s=45,
            alpha=0.75,
            cmap="viridis_r",
            edgecolor="black",
            linewidth=0.35,
            label="Trials",
            zorder=8,
        )

        # Best actual trial
        plt.scatter(
            X_all.loc[best_idx, col_x],
            X_all.loc[best_idx, col_y],
            s=160,
            marker="*",
            color="red",
            edgecolor="black",
            linewidth=1.0,
            label="Best trial",
            zorder=11,
        )

        # 5D continuous KDE optimum projected onto this 2D plane
        # Only plotted when both axes are continuous variables.
        if (
            kde5d_continuous_point is not None
            and variable_info[col_x]["type"] == "continuous"
            and variable_info[col_y]["type"] == "continuous"
            and col_x in kde5d_continuous_point
            and col_y in kde5d_continuous_point
        ):
            plt.scatter(
                kde5d_continuous_point[col_x],
                kde5d_continuous_point[col_y],
                s=120,
                marker="x",
                color="red",
                linewidths=2.5,
                label="5D KDE optimum",
                zorder=12,
            )
        # Smaller KDE peak marker
        plt.scatter(
            peak_x,
            peak_y,
            s=55,
            marker="o",
            color="black",
            edgecolor="white",
            linewidth=1.0,
            label="2D KDE peak",
            zorder=10,
        )

        plt.xlabel(f"{x_label} | bw={bw_x_label}")
        plt.ylabel(f"{y_label} | bw={bw_y_label}")
        plt.title(f"2D KDE: {x_label} vs {y_label}")

        # Proper ticks for discrete/categorical axes
        if variable_info[col_x]["type"] != "continuous":
            plt.xticks(
                variable_info[col_x]["ticks"],
                variable_info[col_x]["ticklabels"],
                rotation=30,
                ha="right",
            )

        if variable_info[col_y]["type"] != "continuous":
            plt.yticks(
                variable_info[col_y]["ticks"],
                variable_info[col_y]["ticklabels"],
            )

        plt.colorbar(cf, label="Kernel regression PDF" if USE_KERNEL_REGRESSION else "Weighted KDE density")
        plt.colorbar(sc, label="Objective validation loss")

        plt.legend()
        plt.tight_layout()

        save_path = output_dir / f"2D_KDE_{safe_filename(x_label)}_vs_{safe_filename(y_label)}.png"

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved 2D KDE plot for {x_label} vs {y_label} to:", save_path)

    results_df = pd.DataFrame(results)

    output_path = csv_path.parent / f"KDE_2D_ALL_VARIABLES_{csv_path.name}"
    results_df.to_csv(output_path, index=False)

    print("All-variable 2D KDE results saved to:", output_path)

    return results_df
# ==================================================
# Overlay kernel sizes onto the 1D KDE plots
# using fixed colours and a normal legend
# ==================================================
def plot_2d_kde_corner(
    df,
    weights,
    csv_path,
    kde_continuous_point=None,
    columns=None,
    grids=120,
    levels=20,
):
    """
    Makes a corner-plot-style grid of 2D KDE projections.

    Diagonal:
    - continuous variables show 1D KDE curves
    - discrete variables show labels only

    Lower triangle:
    - 2D KDE projections
    """

    X_all, variable_info = make_all_kde_variables(df)

    if columns is None:
        columns = list(X_all.columns)

    n_vars = len(columns)
    best_idx = df[loss_col].idxmin()

    # --------------------------------------------------
    # Shared axis limits and ticks for every variable.
    # This keeps the 1D diagonal KDEs and the 2D KDE panels
    # on the same bounds for each variable.
    # --------------------------------------------------
    axis_limits = {}
    axis_ticks = {}

    for c in columns:
        info = variable_info[c]
        dim = space_by_name.get(c)

        if info["type"] == "continuous" and isinstance(dim, (Real, Integer)):
            vmin = float(dim.low)
            vmax = float(dim.high)

            axis_limits[c] = (vmin, vmax)
            axis_ticks[c] = np.linspace(vmin, vmax, 4)

        else:
            ticks = np.asarray(info["ticks"], dtype=float)

            axis_limits[c] = (
                ticks.min(),
                ticks.max(),
            )

            axis_ticks[c] = ticks


    def tick_labels_for(c):
        info = variable_info[c]
        dim = space_by_name.get(c)

        if info["type"] != "continuous":
            return info["ticklabels"]

        if isinstance(dim, Integer):
            return [f"{t:.0f}" for t in axis_ticks[c]]

        labels = []
        for t in axis_ticks[c]:
            if abs(t) < 0.01 and t != 0:
                labels.append(f"{t:.5f}".rstrip("0").rstrip("."))
            else:
                labels.append(f"{t:.2f}")

        return labels

    fig, axes = plt.subplots(
        n_vars,
        n_vars,
        figsize=(2.4 * n_vars, 2.4 * n_vars),
        squeeze=False,
    )

    for row, col_y in enumerate(columns):
        for col, col_x in enumerate(columns):
            ax = axes[row, col]

            # Keep every panel square, whatever the axis ranges or label sizes
            if SQUARE_CORNER_PANELS:
                ax.set_box_aspect(1)

            x_label = variable_info[col_x]["label"]
            y_label = variable_info[col_y]["label"]

            # Hide upper triangle
            if col > row:
                ax.axis("off")
                continue

            # --------------------------------------------------
            # Diagonal: 1D KDE for continuous variables
            # --------------------------------------------------
            if col == row:
                x_diag = X_all[col_x].to_numpy(dtype=float)

                if variable_info[col_x]["type"] == "continuous":
                    grid_1d, pdf_1d = kde_1d(
                        x_diag,
                        weights,
                        grids=grids,
                        var_name=col_x,
                    )

                    bw = kde_bandwidth(x_diag, weights)

                    if bw is not None:
                        bw_label = f"{bw[0]:.3g}"
                    else:
                        bw_label = "NA"

                    if grid_1d is not None:
                        ax.plot(
                            grid_1d,
                            pdf_1d,
                            linewidth=1.5,
                            color="black",
                        )

                        # 1D KDE peak
                        max_idx = np.argmax(pdf_1d)
                        kde_peak = grid_1d[max_idx]
                        peak_density = pdf_1d[max_idx]

                        ax.scatter(
                            kde_peak,
                            peak_density,
                            s=35,
                            marker="o",
                            color="black",
                            edgecolor="white",
                            linewidth=0.6,
                            zorder=10,
                        )

                        # Best actual trial
                        best_x = X_all.loc[best_idx, col_x]
                        best_y = np.interp(best_x, grid_1d, pdf_1d)

                        ax.scatter(
                            best_x,
                            best_y,
                            s=45,
                            marker="*",
                            color="red",
                            edgecolor="black",
                            linewidth=0.5,
                            zorder=11,
                        )

                        # Continuous KDE optimum
                        if (
                            kde_continuous_point is not None
                            and col_x in kde_continuous_point
                        ):
                            opt_x = kde_continuous_point[col_x]
                            opt_y = np.interp(opt_x, grid_1d, pdf_1d)

                            ax.scatter(
                                opt_x,
                                opt_y,
                                s=35,
                                marker="x",
                                color="red",
                                linewidths=1.5,
                                zorder=12,
                            )

                        ax.text(
                            0.05,
                            0.90,
                            f"{x_label}\nbw={bw_label}",
                            ha="left",
                            va="top",
                            fontsize=7,
                            fontweight="bold",
                            transform=ax.transAxes,
                        )

                        ax.set_xlim(axis_limits[col_x])
                        ax.set_xticks(axis_ticks[col_x])
                        ax.set_xticklabels(tick_labels_for(col_x), fontsize=6)
                        ax.set_xlabel(x_label, fontsize=7)
                        ax.set_yticks([])

                    else:
                        ax.text(
                            0.5,
                            0.5,
                            f"{x_label}\nno variation",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontweight="bold",
                            transform=ax.transAxes,
                        )
                        ax.set_xlim(axis_limits[col_x])
                        ax.set_xticks(axis_ticks[col_x])
                        ax.set_xticklabels(tick_labels_for(col_x), fontsize=6)
                        ax.set_yticks([])

                else:
                    ax.text(
                        0.5,
                        0.5,
                        x_label,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                continue

            # --------------------------------------------------
            # Lower triangle: 2D KDE
            # --------------------------------------------------
            x = X_all[col_x].to_numpy(dtype=float)
            y = X_all[col_y].to_numpy(dtype=float)

            X_grid, Y_grid, Z = kde_2d(
                x,
                y,
                weights,
                x_name=col_x,
                y_name=col_y,
                grids=grids,
            )

            if Z is not None:
                ax.contourf(
                    X_grid,
                    Y_grid,
                    Z,
                    levels=levels,
                    cmap="viridis",
                )

                ax.contour(
                    X_grid,
                    Y_grid,
                    Z,
                    levels=levels,
                    colors="black",
                    alpha=0.3,
                    linewidths=0.4,
                )

            # Raw trials
            ax.scatter(
                x,
                y,
                c=df[loss_col],
                s=12,
                alpha=0.65,
                cmap="viridis_r",
                edgecolor="none",
            )

            # Best actual trial
            ax.scatter(
                X_all.loc[best_idx, col_x],
                X_all.loc[best_idx, col_y],
                s=55,
                marker="*",
                color="red",
                edgecolor="black",
                linewidth=0.5,
                zorder=10,
            )

            # Continuous KDE optimum, only if both axes are continuous
            if (
                kde_continuous_point is not None
                and variable_info[col_x]["type"] == "continuous"
                and variable_info[col_y]["type"] == "continuous"
                and col_x in kde_continuous_point
                and col_y in kde_continuous_point
            ):
                ax.scatter(
                    kde_continuous_point[col_x],
                    kde_continuous_point[col_y],
                    s=45,
                    marker="x",
                    color="red",
                    linewidths=1.5,
                    zorder=11,
                )

            ax.set_xlim(axis_limits[col_x])
            ax.set_xticks(axis_ticks[col_x])

            ax.set_ylim(axis_limits[col_y])
            ax.set_yticks(axis_ticks[col_y])

            # Only label outside axes, otherwise it gets too crowded
            if row == n_vars - 1:
                ax.set_xlabel(x_label, fontsize=8)
                ax.set_xticklabels(
                    tick_labels_for(col_x),
                    rotation=45 if variable_info[col_x]["type"] != "continuous" else 0,
                    ha="right" if variable_info[col_x]["type"] != "continuous" else "center",
                    fontsize=7,
                )
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(y_label, fontsize=8)
                ax.set_yticklabels(tick_labels_for(col_y), fontsize=7)
            else:
                ax.set_yticklabels([])

    estimator = "kernel regression" if USE_KERNEL_REGRESSION else "weighted KDE"
    fig.suptitle(
        f"2D KDE corner plot ({estimator}, T = {T:.3g}, "
        f"n_eff = {effective_sample_size(weights):.0f})",
        fontsize=16,
    )
    fig.tight_layout()

    save_path = output_dir / f"2D_KDE_corner_plot_{csv_path.stem}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    print("2D KDE corner plot saved to:", save_path)

    # Left open so it can be shown at the end of the run
    return fig
# ==================================================
# Overlay kernel sizes onto the 1D KDE plots
# using fixed colours and a normal legend
# ==================================================
def plot_kernel_sizes_on_1d_kde(samples, weights, labels, df, csv_path, grids=400):
    """
    For each continuous 1D KDE plot, overlays trial points.

    The point colour is chosen by an if/elif style mapping, for example:
    if ksz_psf == 3, colour purple.

    This is done separately for:
    - ksz_psf
    - ksz_wf
    """

    kernel_cols = [
        col for col in ["ksz_psf", "ksz_wf"]
        if col in df.columns
    ]

    if len(kernel_cols) == 0:
        print("Skipping kernel grouped 1D KDE plots: no kernel size columns found in CSV.")
        return

    def kernel_colour(value):
        """
        Manual colour map for kernel sizes.
        Change these colours if you want different ones.
        """

        value = int(value)

        if value == 1:
            return "tab:blue"
        elif value == 3:
            return "purple"
        elif value == 5:
            return "tab:green"
        elif value == 7:
            return "tab:orange"
        elif value == 9:
            return "tab:red"
        elif value == 11:
            return "tab:brown"
        else:
            return "black"

    for kernel_col in kernel_cols:
        kernel_values = pd.to_numeric(df[kernel_col], errors="coerce").to_numpy(dtype=float)

        if np.isnan(kernel_values).all():
            print(f"Skipping kernel overlay for {kernel_col}: values are not numeric.")
            continue

        unique_kernel_values = np.sort(np.unique(kernel_values[~np.isnan(kernel_values)]))

        for k in range(samples.shape[1]):
            x = samples[:, k]

            grid, pdf = kde_1d(
                x,
                weights,
                grids=grids,
                var_name=labels[k],
            )

            if grid is None:
                print(f"Skipping kernel overlay for {labels[k]} coloured by {kernel_col}: not enough variation.")
                continue

            max_idx = np.argmax(pdf)
            kde_peak = grid[max_idx]
            peak_density = pdf[max_idx]

            # Put each trial point on the KDE curve
            y_on_curve = np.interp(x, grid, pdf)

            plt.figure(figsize=(7, 4.8))

            plt.plot(
                grid,
                pdf,
                linewidth=2,
                color="black",
                label="1D weighted KDE",
            )

            # Plot each kernel size separately so the legend is clear
            for kernel_value in unique_kernel_values:
                mask = kernel_values == kernel_value

                if not np.any(mask):
                    continue

                kernel_label = (
                    str(int(kernel_value))
                    if float(kernel_value).is_integer()
                    else f"{kernel_value:g}"
                )

                plt.scatter(
                    x[mask],
                    y_on_curve[mask],
                    s=55,
                    alpha=0.85,
                    color=kernel_colour(kernel_value),
                    edgecolor="black",
                    linewidth=0.35,
                    label=f"{kernel_col} = {kernel_label}",
                    zorder=8,
                )

            # Smaller 1D KDE peak marker
            plt.scatter(
                kde_peak,
                peak_density,
                s=35,
                color="black",
                edgecolor="white",
                linewidth=0.8,
                zorder=10,
                label="1D KDE peak",
            )

            plt.xlabel(labels[k])
            plt.ylabel("Marginal PDF")
            plt.title(f"1D KDE: {labels[k]} grouped by {kernel_col}")

            plt.legend()
            plt.tight_layout()

            filename = f"1D_KDE_{safe_filename(labels[k])}_grouped_by_{kernel_col}.png"
            save_path = output_dir / filename

            plt.savefig(save_path, dpi=300)
            plt.close()

            print(f"Saved 1D KDE kernel grouped plot for {labels[k]} by {kernel_col} to:", save_path)
# ==================================================
# Discrete variable effect plots
# ==================================================
def plot_discrete_variable_effects(df, csv_path):
    """
    For each discrete variable, show how the objective loss changes
    across its possible values.

    This is useful for variables like:
    - actFunc
    - kernel sizes
    - filter counts
    """

    active_discrete_params = [
        col for col in discrete_params
        if col in df.columns
    ]

    if len(active_discrete_params) == 0:
        print("Skipping discrete variable effects: no discrete parameters found in this CSV.")
        return pd.DataFrame()

    summary_rows = []

    rng = np.random.default_rng(42)

    for col in active_discrete_params:
        grouped_rows = []

        for value, group in df.groupby(col):
            group_weights = group["kde_weight"].to_numpy(dtype=float)
            group_losses = group[loss_col].to_numpy(dtype=float)

            if group_weights.sum() > 0:
                weighted_mean_loss = np.average(group_losses, weights=group_weights)
            else:
                weighted_mean_loss = np.nan

            grouped_rows.append({
                "parameter": col,
                "value": value,
                "count": len(group),
                "mean_loss": group_losses.mean(),
                "median_loss": np.median(group_losses),
                "best_loss": group_losses.min(),
                "weighted_mean_loss": weighted_mean_loss,
            })

        summary = pd.DataFrame(grouped_rows)
        summary = summary.sort_values("mean_loss").reset_index(drop=True)

        summary_rows.append(summary)

        # Category order based on mean loss
        categories = summary["value"].astype(str).tolist()
        positions = np.arange(len(categories))

        # Map each row to x-position
        pos_map = {str(v): i for i, v in enumerate(categories)}
        x_base = df[col].astype(str).map(pos_map).to_numpy(dtype=float)

        # Small jitter so individual trials are visible
        x_jittered = x_base

        plt.figure(figsize=(7, 4.5))

        # Raw trial points
        plt.scatter(
            x_jittered,
            df[loss_col],
            s=45,
            alpha=0.55,
            label="Trials",
        )

        # Mean loss line/points
        plt.plot(
            positions,
            summary["mean_loss"],
            marker="o",
            linewidth=2,
            label="Mean loss",
        )

        # Best loss points
        plt.scatter(
            positions,
            summary["best_loss"],
            s=90,
            marker="*",
            label="Best loss",
            zorder=10,
        )

        plt.xticks(positions, categories)
        plt.xlabel(col)
        plt.ylabel("Objective validation loss")
        plt.title(f"Variable effect: {col}")
        plt.legend()
        plt.tight_layout()

        save_path = output_dir / f"variable_effect_{col}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved variable effect plot for {col} to:", save_path)

    discrete_summary_df = pd.concat(summary_rows, ignore_index=True)

    summary_output_path = csv_path.parent / f"DISCRETE_VARIABLE_EFFECTS_{csv_path.name}"
    discrete_summary_df.to_csv(summary_output_path, index=False)

    print("Discrete variable summary saved to:", summary_output_path)

    return discrete_summary_df


# ==================================================
# Continuous variable effect scatter plots
# ==================================================
def plot_continuous_variable_effects(df, csv_path):
    """
    Simple loss-vs-variable plots for continuous variables.

    These are not KDE plots. They are direct trial scatter plots.
    The variables are taken automatically from optimiser space.
    """

    df_plot = df.copy()
    best_idx = df_plot[loss_col].idxmin()

    for col in continuous_params:
        label = col

        plt.figure(figsize=(6, 4.5))

        plt.scatter(
            df_plot[col].astype(float),
            df_plot[loss_col],
            s=60,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )

        plt.scatter(
            float(df_plot.loc[best_idx, col]),
            df_plot.loc[best_idx, loss_col],
            s=160,
            marker="*",
            color="red",
            edgecolor="black",
            linewidth=1.0,
            label="Best trial",
            zorder=10,
        )

        plt.xlabel(label)
        plt.ylabel("Objective validation loss")
        plt.title(f"Continuous variable effect: {label}")
        plt.legend()
        plt.tight_layout()

        filename = f"continuous_effect_{safe_filename(label)}.png"
        save_path = output_dir / filename

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved continuous effect plot for {label} to:", save_path)


# ==================================================
# Best trial summary
# ==================================================
def save_best_trial_summary(df, csv_path):
    best_idx = df[loss_col].idxmin()
    best_row = df.loc[[best_idx], ["trial", loss_col] + all_params].copy()

    output_path = csv_path.parent / f"BEST_TRIAL_{csv_path.name}"
    best_row.to_csv(output_path, index=False)

    print("\nBest trial:")
    print(best_row.to_string(index=False))
    print("Best trial saved to:", output_path)

def make_continuous_optimum_dict(optimum_transformed):
    if optimum_transformed is None:
        return None

    return {
        name: optimum_transformed[i]
        for i, name in enumerate(continuous_params)
    }


# ==================================================
# Run everything
# ==================================================
if __name__ == "__main__":

    # ------------------------------
    # Continuous 5D KDE optimum
    # Always computed: it is marked on the corner plot and 2D plots.
    # ------------------------------
    optimum_5d_transformed, optimum_5d_physical = find_5d_kde_optimum(
        df=df,
        loss=loss,
        csv_path=csv_path,
        T=T,
    )
    kde_continuous_point = make_continuous_optimum_dict(optimum_5d_transformed)

    # ------------------------------
    # 1D continuous KDE plots
    # ------------------------------
    if MAKE_1D_KDE_PLOTS:
        plot_kde_1d(
            samples=samples,
            weights=weights,
            labels=continuous_labels,
            csv_path=csv_path,
        )

    # ------------------------------
    # All 2D KDE projections
    # Includes:
    # continuous vs continuous
    # continuous vs discrete
    # discrete vs discrete
    # ------------------------------
    if MAKE_2D_KDE_PLOTS:
        all_2d_kde_df = plot_all_2d_kde_pairs(
            df=df,
            weights=weights,
            csv_path=csv_path,
            kde5d_continuous_point=kde_continuous_point,
        )

    # ------------------------------
    # 1D KDE plots grouped by kernel size
    # Uses legend colours instead of colour bar
    # ------------------------------
    if MAKE_KERNEL_GROUPED_1D_PLOTS:
        plot_kernel_sizes_on_1d_kde(
            samples=samples,
            weights=weights,
            labels=continuous_labels,
            df=df,
            csv_path=csv_path,
        )

    # ------------------------------
    # Discrete variable effect plots
    # ------------------------------
    if MAKE_DISCRETE_EFFECT_PLOTS:
        discrete_summary_df = plot_discrete_variable_effects(
            df=df,
            csv_path=csv_path,
        )

    # ------------------------------
    # Continuous variable effect scatter plots
    # ------------------------------
    if MAKE_CONTINUOUS_EFFECT_PLOTS:
        plot_continuous_variable_effects(
            df=df,
            csv_path=csv_path,
        )

    # ------------------------------
    # 2D KDE corner plot
    # ------------------------------
    corner_fig = None

    if MAKE_CORNER_PLOT:
        corner_fig = plot_2d_kde_corner(
            df=df,
            weights=weights,
            csv_path=csv_path,
            kde_continuous_point=kde_continuous_point,
        )

        if not SHOW_CORNER_PLOT:
            plt.close(corner_fig)
            corner_fig = None

    # ------------------------------
    # Best trial summary
    # ------------------------------
    if SAVE_BEST_TRIAL_SUMMARY:
        save_best_trial_summary(
            df=df,
            csv_path=csv_path,
        )

    print("\nAll analysis complete.")

    # ------------------------------
    # Show the corner plot, left open after the script finishes
    # ------------------------------
    if corner_fig is not None:
        plt.ion()
        plt.show(block=False)
