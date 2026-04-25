import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from itertools import combinations

df = pd.read_csv("bayesopt_20260422-1157_all_trials.csv")

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

# Log-transform parameters with large scale differences
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

# Convert loss into weights
T = 0.05
weights = np.exp(-(loss - loss.min()) / T)
weights /= weights.sum()

def kde_2d(x, y, w, grids=200, bw_method=None):
    data = np.vstack([x, y])
    kde = gaussian_kde(data, weights=w, bw_method=bw_method)

    xg = np.linspace(x.min(), x.max(), grids)
    yg = np.linspace(y.min(), y.max(), grids)

    X, Y = np.meshgrid(xg, yg)
    XY = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(XY).reshape(X.shape)

    return X, Y, Z

def plot_kde_pairs(samples, weights, labels, grids=200, levels=30):
    n_params = samples.shape[1]

    for i, j in combinations(range(n_params), 2):
        X, Y, Z = kde_2d(samples[:, i], samples[:, j], weights, grids=grids)

        plt.figure(figsize=(6, 5))

        cf = plt.contourf(X, Y, Z, levels=levels, cmap="viridis")
        plt.contour(X, Y, Z, levels=levels, colors="k", alpha=0.35, linewidths=0.5)

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
            zorder=10
        )

        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.title(f"KDE: {labels[i]} vs {labels[j]}")
        plt.colorbar(cf, label="Weighted KDE density")
        plt.legend()
        plt.tight_layout()
        plt.show()
