from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# --------------------------------------------------
# File path
# --------------------------------------------------
DATA_DIR = Path("/home/manav/PL-NN-testdata_forDec2025/")
DEFAULT_CSV = "bayesopt_20260422-1157_all_trials.csv"

csv_path = DATA_DIR / DEFAULT_CSV
df = pd.read_csv(csv_path)

print("Loaded:", csv_path)


# --------------------------------------------------
# Parameters to analyse
# --------------------------------------------------
params = [
    "learningRate",
    "dropout_rate",
    "dropout_rate_dense",
    "dropout_rate_psf",
    "loss_weight",
    "n_units_dense",
]

loss_col = "objective_val_loss"

samples = df[params].to_numpy(dtype=float)
loss = df[loss_col].to_numpy(dtype=float)


# --------------------------------------------------
# Transform large-scale parameters
# --------------------------------------------------
samples[:, 0] = np.log10(samples[:, 0])      # learningRate
samples[:, 5] = np.log10(samples[:, 5])      # n_units_dense

labels = [
    "log10(learningRate)",
    "dropout_rate",
    "dropout_rate_dense",
    "dropout_rate_psf",
    "loss_weight",
    "log10(n_units_dense)",
]


# --------------------------------------------------
# Convert loss into KDE weights
# Lower loss = better
# --------------------------------------------------
T = 0.05

weights = np.exp(-(loss - loss.min()) / T)
weights /= weights.sum()


# --------------------------------------------------
# 2D KDE function
# --------------------------------------------------
def kde_2d(x, y, w, grids=200, bw_method=None):
    data = np.vstack([x, y])
    kde = gaussian_kde(data, weights=w, bw_method=bw_method)

    xg = np.linspace(x.min(), x.max(), grids)
    yg = np.linspace(y.min(), y.max(), grids)

    X, Y = np.meshgrid(xg, yg)
    XY = np.vstack([X.ravel(), Y.ravel()])

    Z = kde(XY).reshape(X.shape)

    return X, Y, Z


# --------------------------------------------------
# Plot and save KDE graphs
# --------------------------------------------------
def plot_kde_pairs(samples, weights, labels, csv_path, grids=200, levels=30):
    output_dir = csv_path.parent / f"KDE_{csv_path.stem}_plots"
    output_dir.mkdir(exist_ok=True)

    for i, j in combinations(range(samples.shape[1]), 2):
        X, Y, Z = kde_2d(samples[:, i], samples[:, j], weights, grids=grids)

        plt.figure(figsize=(6, 5))

        cf = plt.contourf(X, Y, Z, levels=levels, cmap="viridis")
        plt.contour(
            X,
            Y,
            Z,
            levels=levels,
            colors="k",
            alpha=0.35,
            linewidths=0.5,
        )

        plt.scatter(samples[:, i], samples[:, j], s=25, alpha=0.45)

        # KDE peak
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
            label="KDE peak",
            zorder=10,
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

    print("KDE plots saved to:", output_dir)


# --------------------------------------------------
# Save KDE peak numerical results
# --------------------------------------------------
def save_kde_results(samples, weights, labels, csv_path, grids=200):
    results = []

    for i, j in combinations(range(samples.shape[1]), 2):
        X, Y, Z = kde_2d(samples[:, i], samples[:, j], weights, grids=grids)

        max_idx = np.unravel_index(np.argmax(Z), Z.shape)

        results.append({
            "param_x": labels[i],
            "param_y": labels[j],
            "kde_peak_x": X[max_idx],
            "kde_peak_y": Y[max_idx],
            "peak_density": Z[max_idx],
        })

    results_df = pd.DataFrame(results)

    output_path = csv_path.parent / f"KDE_{csv_path.name}"
    results_df.to_csv(output_path, index=False)

    print("KDE results saved to:", output_path)


# --------------------------------------------------
# Run everything
# --------------------------------------------------
plot_kde_pairs(samples, weights, labels, csv_path)
save_kde_results(samples, weights, labels, csv_path)
