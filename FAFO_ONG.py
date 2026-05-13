from lanternfiber import lanternfiber
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
from scipy.ndimage import zoom


# --------------------------------------------------
# Dataset paths
# --------------------------------------------------
datadir = "/home/manav//PL-NN-testdata_forDec2025/"
slmdatadir = datadir
outdir = datadir

load_precombined_PLims_filename = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined"
load_precombined_wfims_filename = "slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined"

use_subset = 10000
testdatasplit = 0.2
stat_frms = 1000


# --------------------------------------------------
# Load target image from dataset
# --------------------------------------------------
plsavefname = load_precombined_PLims_filename + ".npz"
npf = np.load(datadir + plsavefname, allow_pickle=True)

all_plims = npf["all_plims"]

if use_subset is not None:
    all_plims = all_plims[:use_subset]

# Normalise same way as NN code
mn = np.mean(all_plims[:stat_frms])
sd = np.std(all_plims[:stat_frms])
all_plims = (all_plims - mn) / sd

ndata = all_plims.shape[0]
n_testdata = int(ndata * testdatasplit)

X_test = all_plims[:n_testdata]

target_image = np.squeeze(X_test[0])
target_image = target_image.astype(float)

# Shift to positive for intensity fitting
target_image = target_image - np.min(target_image)
target_image = target_image / np.max(target_image)

print("Target image shape:", target_image.shape)


# --------------------------------------------------
# Generate LP modes
# --------------------------------------------------
n_core = 1.44
n_cladding = 1.4345
wavelength = 1.55
core_radius = 32.8 / 2

N_MODES = 10
N_SAMPLES = 5000
N_COMPONENTS = 10

f = lanternfiber(n_core, n_cladding, core_radius, wavelength)
f.find_fiber_modes()
f.make_fiber_modes(show_plots=False)

all_fields = np.array(f.allmodefields_rsoftorder)

nmodes_total, ny, nx = all_fields.shape
print("Total LP scalar modes:", nmodes_total)
print("LP mode field shape:", ny, nx)

all_fields_10 = all_fields[:N_MODES]

mode_matrix_10 = all_fields_10.reshape(N_MODES, ny * nx).T
mode_matrix_10 = mode_matrix_10.astype(np.complex128)

print("Mode matrix shape:", mode_matrix_10.shape)


# --------------------------------------------------
# Resize target image if needed
# --------------------------------------------------
if target_image.shape != (ny, nx):
    zoom_y = ny / target_image.shape[0]
    zoom_x = nx / target_image.shape[1]

    target_image_resized = zoom(target_image, (zoom_y, zoom_x), order=1)
    target_image_resized = target_image_resized / np.max(target_image_resized)
else:
    target_image_resized = target_image

print("Resized target shape:", target_image_resized.shape)


# --------------------------------------------------
# Least-squares fit of LP coefficients to target intensity
# --------------------------------------------------
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
        intensity = np.abs(field_flat[:, 0]) ** 2

        max_val = np.max(intensity)
        if max_val > 0:
            intensity = intensity / max_val

        return intensity - target_flat

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

    intensity_fit = np.abs(field) ** 2
    intensity_fit = intensity_fit / np.max(intensity_fit)

    return coeffs_fit, intensity_fit, field, res


coeffs_fit, intensity_fit, field_fit, res = fit_lp_coeffs_to_target_intensity(
    mode_matrix_10,
    target_image_resized,
    ny,
    nx,
    max_nfev=2000,
)


# --------------------------------------------------
# Error metrics
# --------------------------------------------------
pca_mse = np.mean((target_pca_recon - target_image_resized) ** 2)
ls_mse = np.mean((intensity_fit - target_image_resized) ** 2)

print("PCA MSE:", pca_mse)
print("Least-squares LP fit MSE:", ls_mse)


# --------------------------------------------------
# Plot comparison
# --------------------------------------------------

plt.imshow(intensity_fit, cmap="inferno")
plt.title(f"Least-squares LP fit\nMSE={ls_mse:.4g}")
plt.colorbar()

plt.tight_layout()

save_path = outdir + f"PCA_vs_LPfit_{N_MODES}modes_target_Xtest0.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print("Saved comparison image to:", save_path)

plt.show()
