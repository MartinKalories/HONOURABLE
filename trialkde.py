import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------
# Inputs: your real data here
# ---------------------------
# samples: shape (N, 3)  -> columns are the 3 parameters
# goodness:     shape (N,) -> this could be a loss value, a gooodness of fit, etc.
# Replace these two lines with your real arrays!

# For the example, make some random data.
N = 100
seed = 0
rng = np.random.default_rng(seed)
# Mixture of 3 correlated multivariate Gaussians (blobby, not spherical)
_components = [
    dict(frac=0.50, mean=[ 0.5, -0.3,  0.8],
         cov=[[ 1.0,  0.6, -0.3],
              [ 0.6,  1.0,  0.1],
              [-0.3,  0.1,  0.8]]),
    dict(frac=0.30, mean=[-1.5,  1.5, -0.5],
         cov=[[ 0.5, -0.35, 0.15],
              [-0.35, 0.7,  0.05],
              [ 0.15, 0.05, 0.45]]),
    dict(frac=0.20, mean=[ 2.2,  0.4,  1.6],
         cov=[[ 0.35, 0.18, 0.22],
              [ 0.18, 0.45,-0.08],
              [ 0.22,-0.08, 0.55]]),
]
_parts = [
    rng.multivariate_normal(c['mean'], c['cov'], size=int(N * c['frac']))
    for c in _components
]
samples = np.vstack(_parts) / 4
rng.shuffle(samples)  # mix blob labels
goodness = np.sum(samples**2, axis=1)  # dummy goodness just for demo




# ---------------------------
# Turn the 'goodness' into weights.
# ---------------------------
# If you had an actual chi-squared goodness of fit, you would do this:
# weights = np.exp(-0.5 * chi2)
# weights /= weights.sum()  # normalize to sum to 1

# But in your case you are just optimising for a 'best', but actual loss value is arbitrary.
# Instead, make weights from some 'goodness' value

weights = np.exp(-0.5 * goodness)
weights /= weights.sum()  # normalize to sum to 1

T = 1 # Temperature  - controls how spikey vs smooth the KDE is
weights = np.exp(-(goodness - goodness.min())/T)
weights /= weights.sum()



# ---------------------------
# Corner plot (if you have 'corner')
# ---------------------------
try:
    import corner
    labels = [r"$\theta_0$", r"$\theta_1$", r"$\theta_2$"]
    fig = corner.corner(
        samples,              # (N, 3)
        weights=weights,      # normalized
        labels=labels,
        bins=40,
        smooth=1.0,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
    )
    plt.show()
except ImportError:
    print("corner not installed; skipping corner plot")

# ---------------------------
# 1D marginals with KDE
# ---------------------------
def kde_1d(x, w, bw_method=None):
    # gaussian_kde expects (d, N) => here d=1
    kde = gaussian_kde(x[np.newaxis, :], weights=w, bw_method=bw_method)
    grid = np.linspace(x.min(), x.max(), 400)
    pdf = kde(grid[np.newaxis, :])
    # normalize area for pretty plotting (optional)
    pdf /= np.trapz(pdf, grid)
    return grid, pdf

labels = [r"$\theta_0$", r"$\theta_1$", r"$\theta_2$"]
for k in range(3):
    x = samples[:, k]
    grid, pdf = kde_1d(x, weights)
    plt.figure()
    plt.plot(grid, pdf)
    plt.xlabel(labels[k])
    plt.ylabel("Marginal PDF (KDE)")
    plt.title(f"1D KDE: {labels[k]}")
    plt.tight_layout()
    plt.show()

# ---------------------------
# 2D joints with KDE (for pairs)
# ---------------------------
def kde_2d(x, y, w, grids=200, bw_method=None):
    # Stack as (d, N) for gaussian_kde, where d=2
    data = np.vstack([x, y]).T            # (N, 2)
    kde = gaussian_kde(data.T, weights=w, bw_method=bw_method)

    xg = np.linspace(x.min(), x.max(), grids)
    yg = np.linspace(y.min(), y.max(), grids)
    X, Y = np.meshgrid(xg, yg)
    XY = np.vstack([X.ravel(), Y.ravel()])  # (2, grids^2)
    Z = kde(XY).reshape(X.shape)

    # Optional visual normalization (not a true integral)
    Z /= np.trapz(np.trapz(Z, xg), yg)
    return X, Y, Z

pairs = [(0,1), (0,2), (1,2)]
for i, j in pairs:
    X, Y, Z = kde_2d(samples[:, i], samples[:, j], weights)

    plt.figure()
    plt.contour(X, Y, Z, levels=10)
    plt.xlabel(labels[i]); plt.ylabel(labels[j])
    plt.title(f"2D KDE: {labels[i]} vs {labels[j]}")
    plt.tight_layout()
    plt.show()


### Make a pretty contour plot of the 2D KDEs
def plot_weight_surfaces_all_pairs(samples, weights, grids=200, bw_method=None,
                                   cmap="viridis", levels=30):

    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [r"$\theta_0$", r"$\theta_1$", r"$\theta_2$"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)

    for ax, (i, j) in zip(axes, pairs):
        X, Y, Z = kde_2d(samples[:, i], samples[:, j], weights,
                         grids=grids, bw_method=bw_method)

        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        ax.contour(X, Y, Z, levels=levels, colors="k", alpha=0.4, linewidths=0.5)

        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.set_title(f"2D Weighted density (KDE): {labels[i]} vs {labels[j]}")

        plt.colorbar(cf, ax=ax, label="Interpolated density (KDE)")

    fig.tight_layout()
    return fig, axes

plot_weight_surfaces_all_pairs(samples, weights)
plt.show()
