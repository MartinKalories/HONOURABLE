from lanternfiber import lanternfiber
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os


# --------------------------------------------------
# Paths
# --------------------------------------------------
datadir = "/home/manav//PL-NN-testdata_forDec2025/"
outdir = datadir
os.makedirs(outdir, exist_ok=True)


# --------------------------------------------------
# Fibre / LP mode settings
# --------------------------------------------------
n_core = 1.44
n_cladding = 1.4345
wavelength = 1.55
core_radius = 32.8 / 2

N_MODES = 10

# Number of random fields to create
N_RANDOM_FIELDS = 100

# Number of generated fields to test by fitting back
N_TEST = 100

# Least-squares settings
MAX_NFEV = 1000
N_RESTARTS = 2

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)


# --------------------------------------------------
# Generate LP modes
# --------------------------------------------------
f = lanternfiber(n_core, n_cladding, core_radius, wavelength)
f.find_fiber_modes()
f.make_fiber_modes(show_plots=False)

all_fields = np.array(f.allmodefields_rsoftorder, dtype=np.complex128)

nmodes_total, ny, nx = all_fields.shape

print("Total LP scalar modes:", nmodes_total)
print("LP mode field shape:", ny, nx)

if N_MODES > nmodes_total:
    raise ValueError(f"N_MODES={N_MODES} is larger than available modes={nmodes_total}")

lp_fields = all_fields[:N_MODES]

# Shape: pixels x modes
mode_matrix = lp_fields.reshape(N_MODES, ny * nx).T

# Normalise each LP mode so coefficients are comparable
mode_norms = np.sqrt(np.sum(np.abs(mode_matrix) ** 2, axis=0, keepdims=True))
mode_matrix = mode_matrix / (mode_norms + 1e-12)

print("Mode matrix shape:", mode_matrix.shape)


# --------------------------------------------------
# Step 1: Pick random complex coefficients
# real and imag parts both in [-1, 1]
# --------------------------------------------------
coeffs_true = (
    rng.uniform(-1, 1, size=(N_RANDOM_FIELDS, N_MODES))
    + 1j * rng.uniform(-1, 1, size=(N_RANDOM_FIELDS, N_MODES))
)

print("True coefficient array shape:", coeffs_true.shape)


# --------------------------------------------------
# Step 2: Multiply by LP basis and add
# field = sum(c_m * LP_m)
# intensity = |field|^2
# --------------------------------------------------
fields_flat = coeffs_true @ mode_matrix.T
fields = fields_flat.reshape(N_RANDOM_FIELDS, ny, nx)

intensities = np.abs(fields) ** 2

# Normalise each intensity image to max = 1
max_vals = np.max(intensities, axis=(1, 2), keepdims=True)
intensities = intensities / (max_vals + 1e-12)

print("Generated intensity dataset shape:", intensities.shape)


# --------------------------------------------------
# Step 3: Save the generated dataset
# --------------------------------------------------
dataset_save_path = os.path.join(
    outdir,
    f"random_LP_dataset_{N_RANDOM_FIELDS}samples_{N_MODES}modes.npz"
)

np.savez_compressed(
    dataset_save_path,
    intensities=intensities.astype(np.float32),
    coeffs_true_real=coeffs_true.real.astype(np.float32),
    coeffs_true_imag=coeffs_true.imag.astype(np.float32),
    lp_fields=lp_fields.astype(np.complex64),
    ny=ny,
    nx=nx,
    N_MODES=N_MODES,
    N_RANDOM_FIELDS=N_RANDOM_FIELDS,
    RNG_SEED=RNG_SEED,
)

print("Saved random LP dataset to:")
print(dataset_save_path)


# --------------------------------------------------
# Least-squares fitting function
# --------------------------------------------------
def fit_lp_coeffs_to_target_intensity(
    mode_matrix,
    target_image,
    max_nfev=1000,
    n_restarts=2,
    rng=None,
):
    """
    Fits complex LP coefficients so that:

        |sum(c_m * LP_m)|^2

    matches the target intensity image.
    """

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

        return  target_flat - intensity

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
# Coefficient alignment
# --------------------------------------------------
def align_coeffs_to_true(coeffs_fit, coeffs_true):
    """
    Intensity fitting has global phase/amplitude ambiguity.

    This aligns the fitted coefficients to the true coefficients before
    calculating coefficient error.
    """

    denom = np.vdot(coeffs_fit, coeffs_fit)

    if np.abs(denom) < 1e-12:
        return coeffs_fit

    alpha = np.vdot(coeffs_fit, coeffs_true) / denom
    return alpha * coeffs_fit


# --------------------------------------------------
# Step 4: Test fitting / reconstruct each generated field
# --------------------------------------------------
N_TEST = min(N_TEST, N_RANDOM_FIELDS)

intensity_rms_errors = []
coeff_rms_errors = []
costs = []
nfevs = []
success_flags = []

example_target = None
example_fit = None
example_residual = None

for i in range(N_TEST):
    print(f"\nFitting sample {i + 1}/{N_TEST}")

    target_image = intensities[i]
    true_coeff = coeffs_true[i]

    coeff_fit, intensity_fit, field_fit, res = fit_lp_coeffs_to_target_intensity(
        mode_matrix,
        target_image,
        max_nfev=MAX_NFEV,
        n_restarts=N_RESTARTS,
        rng=rng,
    )

    # Image/intensity RMS error
    intensity_rms = np.sqrt(np.mean((target_image - intensity_fit) ** 2))

    # Align coefficients before comparing them
    coeff_fit_aligned = align_coeffs_to_true(coeff_fit, true_coeff)

    coeff_rms = np.sqrt(np.mean(np.abs(true_coeff - coeff_fit_aligned ) ** 2))
    intensity_rms_errors.append(intensity_rms)
    coeff_rms_errors.append(coeff_rms)
    costs.append(res.cost)
    nfevs.append(res.nfev)
    success_flags.append(res.success)

    print("Intensity RMS:", intensity_rms)
    print("Coeff RMS:", coeff_rms)
    print("Optimiser success:", res.success)

    EXAMPLE_SAMPLE = 3

    if i == EXAMPLE_SAMPLE - 1:
        example_target = target_image
        example_fit = intensity_fit
        example_residual =  target_image - intensity_fit 


# --------------------------------------------------
# Step 5: Save test results
# --------------------------------------------------
intensity_rms_errors = np.array(intensity_rms_errors)
coeff_rms_errors = np.array(coeff_rms_errors)
costs = np.array(costs)
nfevs = np.array(nfevs)
success_flags = np.array(success_flags)

results_save_path = os.path.join(
    outdir,
    f"LP_fit_test_results_{N_TEST}tests_{N_MODES}modes.npz"
)

np.savez_compressed(
    results_save_path,
    intensity_rms_errors=intensity_rms_errors,
    coeff_rms_errors=coeff_rms_errors,
    costs=costs,
    nfevs=nfevs,
    success_flags=success_flags,
)

print("\nSaved fit results to:")
print(results_save_path)


# --------------------------------------------------
# Save CSV summary
# --------------------------------------------------
csv_save_path = os.path.join(
    outdir,
    f"LP_fit_test_summary_{N_TEST}tests_{N_MODES}modes.csv"
)

summary = np.column_stack([
    np.arange(N_TEST),
    intensity_rms_errors,
    coeff_rms_errors,
    costs,
    nfevs,
    success_flags.astype(int),
])

np.savetxt(
    csv_save_path,
    summary,
    delimiter=",",
    header="sample,intensity_rms,coeff_rms,nfev,success",
    comments="",
)

print("Saved CSV summary to:")
print(csv_save_path)


# --------------------------------------------------
# Print final averages
# --------------------------------------------------
print("\n==============================")
print("Overall fitting results")
print("==============================")
print("Mean intensity RMS:", np.mean(intensity_rms_errors))
print("Median intensity RMS:", np.median(intensity_rms_errors))
print("Mean coeff RMS:", np.mean(coeff_rms_errors))
print("Median coeff RMS:", np.median(coeff_rms_errors))

print("Success rate:", np.mean(success_flags))


# --------------------------------------------------
# Plot one example reconstruction
# --------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(example_target, cmap="inferno")
plt.title("Generated target")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(example_fit, cmap="inferno")
plt.title("LP least-squares fit")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(example_residual, cmap="bwr")
plt.title("Residual")
plt.colorbar()

plt.tight_layout()

plot_save_path = os.path.join(
    outdir,
    f"LP_random_field_fit_example_same_modes{N_MODES}modes.png"
)

plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")

print("Saved example plot to:")
print(plot_save_path)

plt.show()
