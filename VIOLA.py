from lanternfiber import lanternfiber
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.ndimage import zoom
import os


# --------------------------------------------------
# Paths / dataset settings
# --------------------------------------------------
datadir = "/home/manav//PL-NN-testdata_forDec2025/"
outdir = datadir
os.makedirs(outdir, exist_ok=True)

load_precombined_PLims_filename = (
    "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined"
)
precombined_psf_filename = None

use_subset = None
testdatasplit = 0.2
stat_frms = 1000


# --------------------------------------------------
# LP mode settings
# --------------------------------------------------
n_core = 1.44
n_cladding = 1.4345
wavelength = 1.55
core_radius = 32.8 / 2

N_MODES = 10
N_TEST = 20

MAX_NFEV = 1000
N_RESTARTS = 3

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


# --------------------------------------------------
# Load normalised PSF data
# --------------------------------------------------
def load_normalised_psfs():
    if precombined_psf_filename is None:
        psfsavefname = load_precombined_PLims_filename + "-PSFs.npz"
    else:
        psfsavefname = precombined_psf_filename + ".npz"

    print("Loading PSF images from:", psfsavefname)

    npf = np.load(datadir + psfsavefname, allow_pickle=True)
    all_psfims = npf["all_psfims"]

    if use_subset is not None:
        all_psfims = all_psfims[:use_subset]

    # Same normalisation as NN code
    psf_mn = np.percentile(all_psfims[:stat_frms], 0.1)
    all_psfims = all_psfims - psf_mn

    psf_mx = np.percentile(all_psfims[:stat_frms], 99.9)
    all_psfims = all_psfims / (psf_mx + 1e-12)

    ndata = all_psfims.shape[0]
    n_testdata = int(ndata * testdatasplit)

    y_test_psf = all_psfims[:n_testdata]

    print("Normalised y_test_psf shape:", y_test_psf.shape)

    return y_test_psf


# --------------------------------------------------
# Generate LP mode matrix
# --------------------------------------------------
def make_lp_mode_matrix():
    f = lanternfiber(n_core, n_cladding, core_radius, wavelength)
    f.find_fiber_modes()
    f.make_fiber_modes(show_plots=False)

    all_fields = np.array(f.allmodefields_rsoftorder, dtype=np.complex128)
    nmodes_total, ny, nx = all_fields.shape

    print("Total LP scalar modes:", nmodes_total)
    print("LP mode image shape:", ny, nx)

    if N_MODES > nmodes_total:
        raise ValueError(f"N_MODES={N_MODES} > available modes={nmodes_total}")

    lp_fields = all_fields[:N_MODES]

    # Shape: pixels x modes
    mode_matrix = lp_fields.reshape(N_MODES, ny * nx).T

    # Normalise each LP mode column
    mode_norms = np.sqrt(np.sum(np.abs(mode_matrix) ** 2, axis=0, keepdims=True))
    mode_matrix = mode_matrix / (mode_norms + 1e-12)

    print("Mode matrix shape:", mode_matrix.shape)

    return mode_matrix, ny, nx


# --------------------------------------------------
# Resize target PSF to LP grid
# --------------------------------------------------
def resize_target_to_lp_grid(target_image, ny, nx):
    target = np.squeeze(target_image).astype(float)

    # Shift to positive because we are fitting intensity
    target = target - np.min(target)
    target = target / (np.max(target) + 1e-12)

    if target.shape != (ny, nx):
        zoom_y = ny / target.shape[0]
        zoom_x = nx / target.shape[1]

        target = zoom(target, (zoom_y, zoom_x), order=1)
        target = target - np.min(target)
        target = target / (np.max(target) + 1e-12)

    return target


# --------------------------------------------------
# Fit LP coefficients to target PSF intensity
# --------------------------------------------------
def fit_lp_coeffs_to_target_intensity(
    mode_matrix,
    target_image,
    ny,
    nx,
    max_nfev=1000,
    n_restarts=3,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    M = mode_matrix.astype(np.complex128)
    nmodes = M.shape[1]

    target = target_image.astype(float)
    target = target / (np.max(target) + 1e-12)
    target_flat = target.reshape(-1)

    def unpack(z):
        real = z[:nmodes]
        imag = z[nmodes:]
        return real + 1j * imag

    def residual(z):
        coeffs = unpack(z)

        field_flat = M @ coeffs
        intensity = np.abs(field_flat) ** 2
        intensity = intensity / (np.max(intensity) + 1e-12)

        return intensity - target_flat

    best_res = None

    for restart in range(n_restarts):
        if restart == 0:
            z0 = np.zeros(2 * nmodes)
            z0[0] = 1.0
        else:
            z0 = rng.uniform(-1, 1, size=2 * nmodes)

        res = least_squares(
            residual,
            z0,
            max_nfev=max_nfev,
            verbose=0,
        )

        if best_res is None or res.cost < best_res.cost:
            best_res = res

    coeffs_fit = unpack(best_res.x)

    field_flat = M @ coeffs_fit
    field = field_flat.reshape(ny, nx)

    intensity_fit = np.abs(field) ** 2
    intensity_fit = intensity_fit / (np.max(intensity_fit) + 1e-12)

    return coeffs_fit, intensity_fit, field, best_res


# --------------------------------------------------
# Main
# --------------------------------------------------
y_test_psf = load_normalised_psfs()
mode_matrix, ny, nx = make_lp_mode_matrix()

N_TEST = min(N_TEST, y_test_psf.shape[0])

test_indices = rng.choice(
    y_test_psf.shape[0],
    size=N_TEST,
    replace=False,
)

rms_errors = []
costs = []
nfevs = []
success_flags = []

example_target = None
example_fit = None
example_residual = None

for i, idx in enumerate(test_indices):
    print(f"\nFitting PSF {i + 1}/{N_TEST}  |  dataset index = {idx}")

    raw_target_psf = y_test_psf[idx]

    # Resize + normalise to LP grid
    target_psf = resize_target_to_lp_grid(raw_target_psf, ny, nx)

    coeffs_fit, intensity_fit, field_fit, res = fit_lp_coeffs_to_target_intensity(
        mode_matrix=mode_matrix,
        target_image=target_psf,
        ny=ny,
        nx=nx,
        max_nfev=MAX_NFEV,
        n_restarts=N_RESTARTS,
        rng=rng,
    )

    rms = np.sqrt(np.mean((intensity_fit - target_psf) ** 2))

    rms_errors.append(rms)
    costs.append(res.cost)
    nfevs.append(res.nfev)
    success_flags.append(res.success)

    print("PSF RMS error:", rms)
    print("Cost:", res.cost)
    print("nfev:", res.nfev)
    print("Success:", res.success)

    if i == 0:
        example_target = target_psf
        example_fit = intensity_fit
        example_residual = intensity_fit - target_psf


# --------------------------------------------------
# Save numerical results
# --------------------------------------------------
rms_errors = np.array(rms_errors)
costs = np.array(costs)
nfevs = np.array(nfevs)
success_flags = np.array(success_flags)

results_path = os.path.join(
    outdir,
    f"PSF_LP_fit_results_{N_TEST}psfs_{N_MODES}modes.npz",
)

np.savez_compressed(
    results_path,
    test_indices=test_indices,
    rms_errors=rms_errors,
    costs=costs,
    nfevs=nfevs,
    success_flags=success_flags,
    N_TEST=N_TEST,
    N_MODES=N_MODES,
)

print("\nSaved results to:", results_path)

csv_path = os.path.join(
    outdir,
    f"PSF_LP_fit_summary_{N_TEST}psfs_{N_MODES}modes.csv",
)

summary = np.column_stack([
    test_indices,
    rms_errors,
    costs,
    nfevs,
    success_flags.astype(int),
])

np.savetxt(
    csv_path,
    summary,
    delimiter=",",
    header="dataset_index,rms_error,cost,nfev,success",
    comments="",
)

print("Saved CSV summary to:", csv_path)


# --------------------------------------------------
# Print averages
# --------------------------------------------------
print("\n==============================")
print("PSF LP fitting results")
print("==============================")
print("Mean RMS:", np.mean(rms_errors))
print("Median RMS:", np.median(rms_errors))
print("Mean cost:", np.mean(costs))
print("Median cost:", np.median(costs))
print("Mean nfev:", np.mean(nfevs))
print("Success rate:", np.mean(success_flags))


# --------------------------------------------------
# Plot one example
# --------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(example_target, cmap="inferno")
plt.title("Target normalised PSF")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(example_fit, cmap="inferno")
plt.title("LP-mode fit")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(example_residual, cmap="bwr")
plt.title("Residual")
plt.colorbar()

plt.tight_layout()

plot_path = os.path.join(
    outdir,
    f"PSF_LP_fit_example_{N_MODES}modes.png",
)

plt.savefig(plot_path, dpi=300, bbox_inches="tight")

print("Saved example plot to:", plot_path)

plt.show()
