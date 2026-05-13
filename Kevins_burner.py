from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.optimize import differential_evolution

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==================================================
# File path
# ==================================================
DATA_DIR = Path("/home/manav/PL-NN-testdata_forDec2025/")
DEFAULT_CSV = "bayesopt_current_10ksub_noLW_more_epochs_all_trials.csv"

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
    "learningRate",
    "dropout_rate",
    "dropout_rate_dense",
    "dropout_rate_psf",
    "n_units_dense",
]

discrete_params = [
    "ksz_enc",
    "ksz_psf",
    "ksz_wf",
    "nfilts_enc",
    "nfilts_psf",
    "nfilts_wf",
    "actFunc",
]

all_params = continuous_params + discrete_params

continuous_labels = [
    "log10(learningRate)",
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
    """
    Creates the continuous data matrix used for KDE.

    learningRate and n_units_dense are log-transformed because they span
    large numerical ranges.
    """
    samples = df[continuous_params].to_numpy(dtype=float)

    samples[:, 0] = np.log10(samples[:, 0])  # learningRate
    samples[:, 4] = np.log10(samples[:, 4])  # n_units_dense

    return samples


samples = make_continuous_samples(df)


# ==================================================
# KDE helper functions
# ==================================================
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
        x = np.asarray(x).reshape(5, 1)
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
        "learningRate": 10 ** optimum_5d_transformed[0],
        "dropout_rate": optimum_5d_transformed[1],
        "dropout_rate_dense": optimum_5d_transformed[2],
        "dropout_rate_psf": optimum_5d_transformed[3],
        "n_units_dense": 10 ** optimum_5d_transformed[4],
        "log10_learningRate": optimum_5d_transformed[0],
        "log10_n_units_dense": optimum_5d_transformed[4],
        "kde_peak_density_5d": peak_density_5d,
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
# Mixed-variable PCA
# ==================================================
def run_mixed_variable_pca(df, csv_path):
    """
    Builds a 2D PCA map using all variables:
    - continuous variables
    - discrete variables
    - activation function

    Discrete variables are one-hot encoded.
    """

    df_mixed = df[["trial", loss_col] + all_params].copy()

    # Log-transform large-scale continuous variables
    X_continuous = df_mixed[continuous_params].copy()
    X_continuous["learningRate"] = np.log10(X_continuous["learningRate"])
    X_continuous["n_units_dense"] = np.log10(X_continuous["n_units_dense"])

    # Rename transformed columns for clarity
    X_continuous = X_continuous.rename(
        columns={
            "learningRate": "log10_learningRate",
            "n_units_dense": "log10_n_units_dense",
        }
    )

    # One-hot encode discrete variables
    # This is important because actFunc has no natural order.
    X_discrete = pd.get_dummies(
        df_mixed[discrete_params].astype(str),
        prefix=discrete_params,
        dtype=float,
    )

    # Combine continuous and one-hot encoded variables
    X_all = pd.concat([X_continuous, X_discrete], axis=1)

    # Scale before PCA so large-valued columns do not dominate
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "trial": df_mixed["trial"].values,
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        loss_col: df_mixed[loss_col].values,
    })

    # Add original variables back into the output CSV
    for col in all_params:
        pca_df[col] = df_mixed[col].values

    # Save PCA coordinates
    pca_output_path = csv_path.parent / f"MIXED_SPACE_PCA_{csv_path.name}"
    pca_df.to_csv(pca_output_path, index=False)

    # Save PCA loadings so you can interpret what PC1 and PC2 mean
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X_all.columns,
        columns=["PC1_loading", "PC2_loading"],
    )

    loadings_df["abs_PC1_loading"] = loadings_df["PC1_loading"].abs()
    loadings_df["abs_PC2_loading"] = loadings_df["PC2_loading"].abs()

    loadings_output_path = csv_path.parent / f"MIXED_SPACE_PCA_LOADINGS_{csv_path.name}"
    loadings_df.to_csv(loadings_output_path)

    print("Mixed-variable PCA coordinates saved to:", pca_output_path)
    print("Mixed-variable PCA loadings saved to:", loadings_output_path)

    # Plot PCA space
    plt.figure(figsize=(7, 6))

    sc = plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df[loss_col],
        s=80,
        alpha=0.85,
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.4,
    )

    # Mark best trial
    best_idx = pca_df[loss_col].idxmin()

    plt.scatter(
        pca_df.loc[best_idx, "PC1"],
        pca_df.loc[best_idx, "PC2"],
        s=220,
        marker="*",
        color="red",
        edgecolor="black",
        linewidth=1.0,
        label="Best trial",
        zorder=10,
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title("Mixed-variable optimisation space\ncontinuous + discrete variables")
    plt.colorbar(sc, label="Objective validation loss")
    plt.legend()
    plt.tight_layout()

    save_path = output_dir / "mixed_variable_pca_space.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Mixed-variable PCA plot saved to:", save_path)

    return pca_df, loadings_df


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
        jitter = rng.normal(0, 0.04, size=len(x_base))
        x_jittered = x_base + jitter

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

    df_plot["log10_learningRate"] = np.log10(df_plot["learningRate"])
    df_plot["log10_n_units_dense"] = np.log10(df_plot["n_units_dense"])

    plot_cols = [
        ("log10_learningRate", "log10(learningRate)"),
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
    # Original continuous KDE section
    # ------------------------------
    optimum_5d_transformed, optimum_5d_physical = find_5d_kde_optimum(
        df=df,
        loss=loss,
        csv_path=csv_path,
        T=T,
    )

    kde5d_point_for_2d = make_5d_point_for_2d_plots(optimum_5d_transformed)

    plot_kde_1d(
        samples=samples,
        weights=weights,
        labels=continuous_labels,
        csv_path=csv_path,
    )

    plot_kde_pairs(
        samples=samples,
        weights=weights,
        labels=continuous_labels,
        csv_path=csv_path,
        kde5d_point=kde5d_point_for_2d,
    )

    save_kde_results(
        samples=samples,
        weights=weights,
        labels=continuous_labels,
        csv_path=csv_path,
        kde5d_point=kde5d_point_for_2d,
    )

    # ------------------------------
    # New mixed-variable analysis
    # ------------------------------
    pca_df, loadings_df = run_mixed_variable_pca(
        df=df,
        csv_path=csv_path,
    )

    discrete_summary_df = plot_discrete_variable_effects(
        df=df,
        csv_path=csv_path,
    )

    plot_continuous_variable_effects(
        df=df,
        csv_path=csv_path,
    )

    save_best_trial_summary(
        df=df,
        csv_path=csv_path,
    )

    print("\nAll analysis complete.")
