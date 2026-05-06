from lanternfiber import lanternfiber
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# --------------------------------------------------
# Fibre parameters
# --------------------------------------------------
n_core = 1.44
n_cladding = 1.4345
wavelength = 1.55      # microns
core_radius = 32.8 / 2 # microns

N_MODES = 10
N_SAMPLES = 5000
N_COMPONENTS = 10


# --------------------------------------------------
# Generate LP modes
# --------------------------------------------------
f = lanternfiber(n_core, n_cladding, core_radius, wavelength)
f.find_fiber_modes()
f.make_fiber_modes(show_plots=False)

all_fields = np.array(f.allmodefields_rsoftorder)

nmodes, ny, nx = all_fields.shape
print("Total modes available:", nmodes)
print("Mode field shape:", ny, nx)

# Use only first 10 modes
all_fields_10 = all_fields[:N_MODES]

# Matrix shape: pixels x modes
mode_matrix_10 = all_fields_10.reshape(N_MODES, ny * nx).T
mode_matrix_10 = mode_matrix_10.astype(np.complex128)

print("Mode matrix shape:", mode_matrix_10.shape)


# --------------------------------------------------
# Generate synthetic intensity images from random coefficients
# --------------------------------------------------
X = []

for _ in range(N_SAMPLES):
    coeffs = np.random.randn(N_MODES, 1) + 1j * np.random.randn(N_MODES, 1)

    # Normalise coefficient power
    coeffs /= np.linalg.norm(coeffs)

    field_flat = mode_matrix_10 @ coeffs
    field = field_flat.reshape(ny, nx)

    intensity = np.abs(field) ** 2

    # Normalise intensity
    intensity /= np.max(intensity)

    X.append(intensity.reshape(-1))

X = np.array(X)

print("Synthetic PCA dataset shape:", X.shape)


# --------------------------------------------------
# Run PCA / eigenfaces-style decomposition
# --------------------------------------------------
pca = PCA(n_components=N_COMPONENTS)
pca.fit(X)

eigenpatterns = pca.components_.reshape(N_COMPONENTS, ny, nx)

print("Explained variance ratios:")
print(pca.explained_variance_ratio_)


# --------------------------------------------------
# Plot eigen-intensity patterns
# --------------------------------------------------
for k in range(N_COMPONENTS):
    plt.figure(figsize=(5, 4))
    plt.imshow(eigenpatterns[k], cmap="bwr")
    plt.colorbar(label="PCA component value")
    plt.title(f"Eigen-intensity pattern {k + 1}")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Example: reconstruct one generated image using PCA
# --------------------------------------------------
sample_idx = 0

original = X[sample_idx]
pca_coeffs = pca.transform(original.reshape(1, -1))
reconstructed = pca.inverse_transform(pca_coeffs)

original_img = original.reshape(ny, nx)
reconstructed_img = reconstructed.reshape(ny, nx)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap="inferno")
plt.title("Original synthetic intensity")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img, cmap="inferno")
plt.title("PCA reconstruction")
plt.colorbar()

plt.tight_layout()

save_path = f"/home/manav/PL-NN-testdata_forDec2025/PCA_{N_MODES}modes_reconstruction.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"Saved comparison image to: {save_path}")
plt.show()


from scipy.optimize import least_squares
import numpy as np

def fit_lp_coeffs_to_target_intensity(mode_matrix, target_image, ny, nx, max_nfev=2000):
    M = mode_matrix.astype(np.complex128)
    nmodes = M.shape[1]

    target = target_image.astype(float)
    target = target / np.max(target)
    target_flat = target.reshape(-1)

    def unpack(z):
        real = z[:nmodes]
        imag = z[nmodes:]
        coeffs = real + 1j * imag
        return coeffs.reshape(nmodes, 1)

    def residual(z):
        coeffs = unpack(z)

        field_flat = M @ coeffs
        intensity = np.abs(field_flat[:, 0])**2

        max_val = np.max(intensity)
        if max_val > 0:
            intensity = intensity / max_val

        return intensity - target_flat

    # Initial guess: mostly LP01
    z0 = np.zeros(2 * nmodes)
    z0[0] = 1.0

    res = least_squares(
        residual,
        z0,
        max_nfev=max_nfev,
        verbose=1,
    )

    coeffs_fit = unpack(res.x)

    field_flat = M @ coeffs_fit
    field = field_flat.reshape(ny, nx)

    intensity_fit = np.abs(field)**2
    intensity_fit = intensity_fit / np.max(intensity_fit)

    return coeffs_fit, intensity_fit, field, res

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(target_image / np.max(target_image), cmap="inferno")
plt.title("Target image")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(intensity_fit, cmap="inferno")
plt.title("Fitted LP intensity")
plt.colorbar()

plt.tight_layout()
plt.show()

