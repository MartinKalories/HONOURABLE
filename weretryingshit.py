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
plt.show()
plt.save()
