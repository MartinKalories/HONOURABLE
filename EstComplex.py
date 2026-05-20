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
# Least-squares fitting function for COMPLEX field amplitudes
# --------------------------------------------------
def fit_lp_coeffs_to_target_complex_field(mode_matrix, target_field):
    """
    Fits complex LP coefficients directly to the complex field amplitude:

        target_field ≈ sum(c_m * LP_m)

    This is a linear complex least-squares problem.
    """

    M = mode_matrix.astype(np.complex128)

    # Flatten target complex field into pixels
    target_flat = target_field.reshape(-1).astype(np.complex128)

    # Solve M @ coeffs_fit ≈ target_flat
    coeffs_fit, residuals, rank, singular_values = np.linalg.lstsq(
        M,
        target_flat,
        rcond=None
    )

    # Reconstruct complex field
    field_fit_flat = M @ coeffs_fit
    field_fit = field_fit_flat.reshape(ny, nx)

    # Reconstructed intensity, just for plotting
    intensity_fit = np.abs(field_fit) ** 2
    intensity_fit = intensity_fit / (np.max(intensity_fit) + 1e-12)

    return coeffs_fit, field_fit, intensity_fit, residuals, rank, singular_values


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
# Step 4: Test fitting complex amplitudes
# --------------------------------------------------
N_TEST = min(N_TEST, N_RANDOM_FIELDS)

field_rms_errors = []
intensity_rms_errors = []
coeff_rms_errors = []
ranks = []

example_target_intensity = None
example_fit_intensity = None
example_field_residual_abs = None

EXAMPLE_SAMPLE = 10

for i in range(N_TEST):
    print(f"\nFitting sample {i + 1}/{N_TEST}")

    # Complex generated field
    target_field = fields[i]

    # Intensity only used for comparison/plotting
    target_intensity = intensities[i]

    # True generated complex coefficients
    true_coeff = coeffs_true[i]

    coeff_fit, field_fit, intensity_fit, residuals, rank, singular_values = (
        fit_lp_coeffs_to_target_complex_field(
            mode_matrix,
            target_field
        )
    )

    # Complex field RMS error
    field_rms = np.sqrt(np.mean(np.abs(target_field - field_fit) ** 2))

    # Intensity RMS error, mostly for visual comparison
    intensity_rms = np.sqrt(np.mean((target_intensity - intensity_fit) ** 2))

    # Since we fit the complex field directly, no phase alignment is needed
    coeff_rms = np.sqrt(np.mean(np.abs(true_coeff - coeff_fit) ** 2))

    field_rms_errors.append(field_rms)
    intensity_rms_errors.append(intensity_rms)
    coeff_rms_errors.append(coeff_rms)
    ranks.append(rank)

    print("Complex field RMS:", field_rms)
    print("Intensity RMS:", intensity_rms)
    print("Coeff RMS:", coeff_rms)
    print("Matrix rank:", rank)

    if i == EXAMPLE_SAMPLE - 1:
        example_target_intensity = target_intensity
        example_fit_intensity = intensity_fit
        example_field_residual_abs = np.abs(target_field - field_fit)
# --------------------------------------------------
# Step 5: Save test results
# --------------------------------------------------
field_rms_errors = np.array(field_rms_errors)
intensity_rms_errors = np.array(intensity_rms_errors)
coeff_rms_errors = np.array(coeff_rms_errors)
ranks = np.array(ranks)

results_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_results_{N_TEST}tests_{N_MODES}modes.npz"
)

np.savez_compressed(
    results_save_path,
    field_rms_errors=field_rms_errors,
    intensity_rms_errors=intensity_rms_errors,
    coeff_rms_errors=coeff_rms_errors,
    ranks=ranks,
)

print("\nSaved complex field fit results to:")
print(results_save_path)

# --------------------------------------------------
# Save CSV summary
# --------------------------------------------------
csv_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_summary_{N_TEST}tests_{N_MODES}modes.csv"
)

summary = np.column_stack([
    np.arange(N_TEST),
    field_rms_errors,
    intensity_rms_errors,
    coeff_rms_errors,
    ranks,
])

np.savetxt(
    csv_save_path,
    summary,
    delimiter=",",
    header="sample,field_rms,intensity_rms,coeff_rms,rank",
    comments="",
)

print("Saved CSV summary to:")
print(csv_save_path)
# --------------------------------------------------
# Print final averages
# --------------------------------------------------
print("\n==============================")
print("Overall complex field fitting results")
print("==============================")
print("Mean complex field RMS:", np.mean(field_rms_errors))
print("Median complex field RMS:", np.median(field_rms_errors))

print("Mean intensity RMS:", np.mean(intensity_rms_errors))
print("Median intensity RMS:", np.median(intensity_rms_errors))

print("Mean coeff RMS:", np.mean(coeff_rms_errors))
print("Median coeff RMS:", np.median(coeff_rms_errors))


# --------------------------------------------------
# Plot one example reconstruction
# --------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(example_target_intensity, cmap="inferno")
plt.title("Target intensity")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(example_fit_intensity, cmap="inferno")
plt.title("Fit intensity from complex LS")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(example_field_residual_abs, cmap="viridis")
plt.title(r"$|E_\mathrm{target} - E_\mathrm{fit}|$")
plt.colorbar()

plt.tight_layout()

plot_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_example{EXAMPLE_SAMPLE}_{N_MODES}modes.png"
)

plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")

print("Saved example plot to:")
print(plot_save_path)

plt.show()
