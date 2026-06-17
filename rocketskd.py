from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.optimize import differential_evolution


# ==================================================
# File path
# ==================================================
DATA_DIR = Path("/home/manav/PL-NN-testdata_forDec2025/")
DEFAULT_CSV = "bayesopt_current_nosub_noLW_more_epochs_all_trials.csv"

csv_path = DATA_DIR / DEFAULT_CSV
df = pd.read_csv(csv_path)

print("Loaded:", csv_path)


# ==================================================
# Settings
# ==================================================
loss_col = "objective_val_loss"

# Lower T makes only the best trials matter strongly.
# Higher T gives smoother weighting across more trials.
T = 0.1

continuous_params = [
    "dropout_rate",
    "dropout_rate_dense",
    "dropout_rate_psf",
    "n_units_dense",
]

discrete_params = [
    "ksz_psf",
    "ksz_wf",
    "nfilts_psf",
    "nfilts_wf",
]

all_params = continuous_params + discrete_params

continuous_labels = [
    "dropout_rate",
    "dropout_rate_dense",
    "dropout_rate_psf",
    "log10(n_units_dense)",
]


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
output_dir = csv_path.parent / f"KDE_{csv_path.stem}_plots"
output_dir.mkdir(exist_ok=True)

print("Plots will be saved to:", output_dir)


# ==================================================
# Convert loss into goodness weights
# Lower loss = better
# ==================================================
goodness = np.exp(-(loss - loss.min()) / T)
weights = goodness / goodness.sum()

df["kde_weight"] = weights


# ==================================================
# Continuous samples for KDE
# ==================================================
def make_continuous_samples(df):
    samples_df = df[continuous_params].astype(float).copy()

    if "learningRate" in samples_df.columns:
        samples_df["learningRate"] = np.log10(samples_df["learningRate"])

    if "n_units_dense" in samples_df.columns:
        samples_df["n_units_dense"] = np.log10(samples_df["n_units_dense"])

    return samples_df.to_numpy(dtype=float)


samples = make_continuous_samples(df)


# ==================================================
# KDE helper functions
# ==================================================
def kde_bandwidth(data, weights=None, bw_method=None):
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

def kde_1d(x, w, grids=400, bw_method=None):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    if np.std(x) < 1e-12:
        return None, None

    try:
        kde = gaussian_kde(
            x[np.newaxis, :],
            weights=w,
            bw_method=bw_method,
        )

        grid = np.linspace(x.min(), x.max(), grids)
        pdf = kde(grid[np.newaxis, :])

        area = trapezoid(pdf, grid)
        if area > 0:
            pdf /= area

        return grid, pdf

    except np.linalg.LinAlgError:
        return None, None


def kde_2d(x, y, w, grids=200, bw_method=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    xg = np.linspace(x.min(), x.max(), grids)
    yg = np.linspace(y.min(), y.max(), grids)

    X, Y = np.meshgrid(xg, yg)

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return X, Y, None

    try:
        data = np.vstack([x, y])

        kde = gaussian_kde(
            data,
            weights=w,
            bw_method=bw_method,
        )

        XY = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(XY).reshape(X.shape)

        return X, Y, Z

    except np.linalg.LinAlgError:
        return X, Y, None
        
def make_continuous_optimum_dict(optimum_transformed):
    if optimum_transformed is None:
        return None

    return {
        "dropout_rate": optimum_transformed[0],
        "dropout_rate_dense": optimum_transformed[1],
        "dropout_rate_psf": optimum_transformed[2],
        "log10_n_units_dense": optimum_transformed[3],
    }


# ==================================================
# Find continuous 5D KDE optimum
# ==================================================
def find_5d_kde_optimum(df, loss, csv_path, T=0.1, bw_method=None):
    samples_5d = make_continuous_samples(df)

    goodness_5d = np.exp(-(loss - loss.min()) / T)
    weights_5d = goodness_5d / goodness_5d.sum()

    try:
        kde = gaussian_kde(
            samples_5d.T,
            weights=weights_5d,
            bw_method=bw_method,
        )
    except np.linalg.LinAlgError:
        print("Could not compute 5D KDE optimum because the covariance matrix was singular.")
        return None, None

    bounds = [
        (samples_5d[:, k].min(), samples_5d[:, k].max())
        for k in range(samples_5d.shape[1])
    ]

    def negative_kde_density(x):
        x = np.asarray(x).reshape(len(continuous_params), 1)
        return -kde(x)[0]

    result = differential_evolution(
        negative_kde_density,
        bounds=bounds,
        seed=42,
        polish=True,
    )

    optimum_5d_transformed = result.x
    peak_density_5d = -result.fun

    optimum_physical = {
        "dropout_rate": optimum_5d_transformed[0],
        "dropout_rate_dense": optimum_5d_transformed[1],
        "dropout_rate_psf": optimum_5d_transformed[2],
        "n_units_dense": 10 ** optimum_5d_transformed[3],
        "log10_n_units_dense": optimum_5d_transformed[3],
        "kde_peak_density_4d": peak_density_5d,
        "T": T,
    }

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

    point_5d = np.full(5, np.nan)

    point_5d[0] = optimum_5d_transformed[0]  # log10 learningRate
    point_5d[1] = optimum_5d_transformed[1]  # dropout_rate
    point_5d[2] = optimum_5d_transformed[2]  # dropout_rate_dense
    point_5d[3] = optimum_5d_transformed[3]  # dropout_rate_psf
    point_5d[4] = optimum_5d_transformed[4]  # log10 n_units_dense

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
            print(f"1D KDE bandwidth for {labels[k]}: {bw[0]:.6g}")

        grid, pdf = kde_1d(x, weights, grids=grids)

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
        plt.title(f"1D KDE: {labels[k]}")
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
    """
    Builds one combined dataframe for 2D KDE plots.

    Continuous variables:
    - learningRate is converted to log10(learningRate)
    - n_units_dense is converted to log10(n_units_dense)

    Discrete variables:
    - numeric discrete variables keep their real values
    - categorical variables like actFunc are encoded as 0, 1, 2...
    """

    X = pd.DataFrame(index=df.index)
    variable_info = {}

    # ------------------------------
    # Continuous variables
    # ------------------------------
    
    X["dropout_rate"] = df["dropout_rate"].astype(float)
    variable_info["dropout_rate"] = {
        "label": "dropout_rate",
        "type": "continuous",
    }

    X["dropout_rate_dense"] = df["dropout_rate_dense"].astype(float)
    variable_info["dropout_rate_dense"] = {
        "label": "dropout_rate_dense",
        "type": "continuous",
    }

    X["dropout_rate_psf"] = df["dropout_rate_psf"].astype(float)
    variable_info["dropout_rate_psf"] = {
        "label": "dropout_rate_psf",
        "type": "continuous",
    }

    X["log10_n_units_dense"] = np.log10(df["n_units_dense"].astype(float))
    variable_info["log10_n_units_dense"] = {
        "label": "log10(n_units_dense)",
        "type": "continuous",
    }

    # ------------------------------
    # Discrete variables
    # ------------------------------
    for col in discrete_params:
        numeric_values = pd.to_numeric(df[col], errors="coerce")

        # Numeric discrete variables, for example ksz_psf = 3, 5, 7
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

        # Categorical discrete variables, for example actFunc
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
        bw = kde_bandwidth(
            np.vstack([x, y]),
            weights=weights,
        )

    if bw is not None:
        print(
            f"2D KDE bandwidth for {x_label} vs {y_label}: "
            f"{x_label} = {bw[0]:.6g}, {y_label} = {bw[1]:.6g}"
        )

        X_grid, Y_grid, Z = kde_2d(
            x,
            y,
            weights,
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

        plt.xlabel(x_label)
        plt.ylabel(y_label)
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

        plt.colorbar(cf, label="Weighted KDE density")
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

    No histograms are plotted on the diagonal. The diagonal only labels
    each variable.
    """

    X_all, variable_info = make_all_kde_variables(df)

    if columns is None:
        columns = list(X_all.columns)

    n_vars = len(columns)
    best_idx = df[loss_col].idxmin()

    fig, axes = plt.subplots(
        n_vars,
        n_vars,
        figsize=(2.4 * n_vars, 2.4 * n_vars),
        squeeze=False,
    )

    for row, col_y in enumerate(columns):
        for col, col_x in enumerate(columns):
            ax = axes[row, col]

            x_label = variable_info[col_x]["label"]
            y_label = variable_info[col_y]["label"]

            # Hide upper triangle
            if col > row:
                ax.axis("off")
                continue

            # Diagonal: just show variable name
            if col == row:
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

            x = X_all[col_x].to_numpy(dtype=float)
            y = X_all[col_y].to_numpy(dtype=float)

            X_grid, Y_grid, Z = kde_2d(
                x,
                y,
                weights,
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

            # Only label outside axes, otherwise it gets too crowded
            if row == n_vars - 1:
                ax.set_xlabel(x_label, fontsize=8)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(y_label, fontsize=8)
            else:
                ax.set_yticklabels([])

            # Proper ticks for discrete axes
            if variable_info[col_x]["type"] != "continuous":
                ax.set_xticks(variable_info[col_x]["ticks"])
                ax.set_xticklabels(
                    variable_info[col_x]["ticklabels"],
                    rotation=45,
                    ha="right",
                    fontsize=7,
                )

            if variable_info[col_y]["type"] != "continuous":
                ax.set_yticks(variable_info[col_y]["ticks"])
                ax.set_yticklabels(
                    variable_info[col_y]["ticklabels"],
                    fontsize=7,
                )

    fig.suptitle("2D KDE corner plot", fontsize=16)
    fig.tight_layout()

    save_path = output_dir / f"2D_KDE_corner_plot_{csv_path.stem}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("2D KDE corner plot saved to:", save_path)
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

    kernel_cols = ["ksz_psf", "ksz_wf"]

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

    summary_rows = []

    rng = np.random.default_rng(42)

    for col in discrete_params:
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
    """

    df_plot = df.copy()

    
    df_plot["log10_n_units_dense"] = np.log10(df_plot["n_units_dense"])

    plot_cols = [
        ("dropout_rate", "dropout_rate"),
        ("dropout_rate_dense", "dropout_rate_dense"),
        ("dropout_rate_psf", "dropout_rate_psf"),
        ("log10_n_units_dense", "log10(n_units_dense)"),
    ]

    for col, label in plot_cols:
        plt.figure(figsize=(6, 4.5))

        plt.scatter(
            df_plot[col],
            df_plot[loss_col],
            s=60,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )

        best_idx = df_plot[loss_col].idxmin()

        plt.scatter(
            df_plot.loc[best_idx, col],
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

        filename = (
            f"continuous_effect_{label}.png"
            .replace("(", "")
            .replace(")", "")
            .replace("/", "")
        )

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


# ==================================================
# Run everything
# ==================================================
if __name__ == "__main__":

    # ------------------------------
    # Continuous 5D KDE optimum
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
    discrete_summary_df = plot_discrete_variable_effects(
        df=df,
        csv_path=csv_path,
    )

    # ------------------------------
    # Continuous variable effect scatter plots
    # ------------------------------
    plot_continuous_variable_effects(
        df=df,
        csv_path=csv_path,
    )
    plot_2d_kde_corner(
        df=df,
        weights=weights,
        csv_path=csv_path,
        kde_continuous_point=kde_continuous_point,
    )
    # ------------------------------
    # Best trial summary
    # ------------------------------
    save_best_trial_summary(
        df=df,
        csv_path=csv_path,
    )

    print("\nAll analysis complete.")
