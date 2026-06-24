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
# Use more modes to generate the target field
N_MODES_GENERATE = 17

# Use fewer modes to fit/reconstruct the target field
N_MODES_FIT = 15

# Number of random fields to create
N_RANDOM_FIELDS = 100

# Number of generated fields to test by fitting back
N_TEST = 100

# Least-squares settings
MAX_NFEV = 1000
N_RESTARTS = 2

RNG_SEED = 41
rng = np.random.default_rng(RNG_SEED)
EXAMPLE_INDEX = 12

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

if N_MODES_GENERATE > nmodes_total:
    raise ValueError(
        f"N_MODES_GENERATE={N_MODES_GENERATE} is larger than available modes={nmodes_total}"
    )

if N_MODES_FIT > N_MODES_GENERATE:
    raise ValueError(
        "N_MODES_FIT should be less than or equal to N_MODES_GENERATE for this test."
    )

# Modes used to generate target fields
lp_fields_generate = all_fields[:N_MODES_GENERATE]

# Modes used by the least-squares fitting model
lp_fields_fit = all_fields[:N_MODES_FIT]


def make_mode_matrix(lp_fields):
    """
    Convert LP mode images into a matrix with shape:

        pixels x modes

    and normalise each mode to unit energy.
    """
    nmodes_local = lp_fields.shape[0]

    M = lp_fields.reshape(nmodes_local, ny * nx).T

    mode_norms = np.sqrt(np.sum(np.abs(M) ** 2, axis=0, keepdims=True))
    M = M / (mode_norms + 1e-12)

    return M


mode_matrix_generate = make_mode_matrix(lp_fields_generate)
mode_matrix_fit = make_mode_matrix(lp_fields_fit)

print("Generation mode matrix shape:", mode_matrix_generate.shape)
print("Fitting mode matrix shape:", mode_matrix_fit.shape)

# --------------------------------------------------
# Step 1: Pick random complex coefficients
# These are the TRUE coefficients used to generate the targets.
# --------------------------------------------------
coeffs_true = (
    rng.uniform(-1, 1, size=(N_RANDOM_FIELDS, N_MODES_GENERATE))
    + 1j * rng.uniform(-1, 1, size=(N_RANDOM_FIELDS, N_MODES_GENERATE))
)

print("True coefficient array shape:", coeffs_true.shape)


# --------------------------------------------------
# Step 2: Generate target complex fields and intensities
# field = sum(c_m * LP_m)
# intensity = |field|^2
# --------------------------------------------------
fields_flat_true = coeffs_true @ mode_matrix_generate.T
fields_true = fields_flat_true.reshape(N_RANDOM_FIELDS, ny, nx)

intensities = np.abs(fields_true) ** 2

# Normalise each intensity image to max = 1
max_vals = np.max(intensities, axis=(1, 2), keepdims=True)
intensities = intensities / (max_vals + 1e-12)

# Normalise the TRUE complex fields in the same way.
# This makes |fields_true_normalised|^2 exactly match the normalised target intensity.
fields_true_normalised = fields_true / np.sqrt(max_vals + 1e-12)

print("Generated intensity dataset shape:", intensities.shape)
# --------------------------------------------------
# Step 3: Save the generated dataset
# --------------------------------------------------
dataset_save_path = os.path.join(
    outdir,
    f"random_LP_dataset_{N_RANDOM_FIELDS}samples_gen{N_MODES_GENERATE}_fit{N_MODES_FIT}modes.npz"
)

np.savez_compressed(
    dataset_save_path,
    intensities=intensities.astype(np.float32),
    fields_true_real=fields_true_normalised.real.astype(np.float32),
    fields_true_imag=fields_true_normalised.imag.astype(np.float32),
    coeffs_true_real=coeffs_true.real.astype(np.float32),
    coeffs_true_imag=coeffs_true.imag.astype(np.float32),
    lp_fields_generate=lp_fields_generate.astype(np.complex64),
    lp_fields_fit=lp_fields_fit.astype(np.complex64),
    ny=ny,
    nx=nx,
    N_MODES_GENERATE=N_MODES_GENERATE,
    N_MODES_FIT=N_MODES_FIT,
    N_RANDOM_FIELDS=N_RANDOM_FIELDS,
    RNG_SEED=RNG_SEED,
)

print("Saved random LP dataset to:")
print(dataset_save_path)

# --------------------------------------------------
# Least-squares fitting function
# --------------------------------------------------
def fit_lp_coeffs_to_target_complex_field(
    mode_matrix,
    target_field,
    max_nfev=1000,
    n_restarts=2,
    rng=None,
):
    """
    Fits complex LP coefficients so that:

        sum(c_m * LP_m)

    matches the target complex field.

    scipy.optimize.least_squares only accepts real-valued residuals, so the
    complex field error is split into real and imaginary parts.
    """

    if rng is None:
        rng = np.random.default_rng()

    M = mode_matrix.astype(np.complex128)
    nmodes = M.shape[1]

    target = target_field.astype(np.complex128)
    target = normalise_complex_field_to_unit_intensity(target)
    target_flat = target.reshape(-1)

    def unpack(z):
        real = z[:nmodes]
        imag = z[nmodes:]
        return real + 1j * imag

    def residual(z):
        coeffs = unpack(z)

        field_flat = M @ coeffs
        complex_error = field_flat - target_flat

        return np.concatenate([
            complex_error.real,
            complex_error.imag,
        ])

    best_res = None

    for restart in range(n_restarts):
        if restart == 0:
            coeffs0, *_ = np.linalg.lstsq(M, target_flat, rcond=None)
            z0 = np.concatenate([coeffs0.real, coeffs0.imag])
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
# Complex-field comparison helpers
# --------------------------------------------------
def normalise_complex_field_to_unit_intensity(field):
    """
    Scale a complex field so that max(|E|^2) = 1.

    This is needed because the intensity fitting function returns a raw
    complex field, while the plotted intensity is normalised to max = 1.
    """
    max_intensity = np.max(np.abs(field) ** 2)
    return field / (np.sqrt(max_intensity) + 1e-12)


def align_field_global_phase(reference_field, field_to_align, amp_threshold=0.00):
    """
    Align the fitted field to the target field by one global phase factor.

    Intensity fitting only sees |E|^2, so it cannot determine the absolute
    global phase of E. Without this alignment, |E_target - E_fit| can look
    artificially large even when the intensity fit is good.

    The mask avoids using near-zero-amplitude pixels where phase is noisy.
    """
    ref_flat = reference_field.reshape(-1)
    fit_flat = field_to_align.reshape(-1)

    ref_amp = np.abs(ref_flat)
    fit_amp = np.abs(fit_flat)

    mask = (
        (ref_amp > amp_threshold * np.max(ref_amp)) &
        (fit_amp > amp_threshold * np.max(fit_amp))
    )

    if np.sum(mask) < 10:
        mask = np.ones_like(ref_amp, dtype=bool)

    overlap = np.vdot(fit_flat[mask], ref_flat[mask])

    if np.abs(overlap) < 1e-12:
        return field_to_align, 1.0 + 0.0j

    phase_factor = overlap / np.abs(overlap)

    return phase_factor * field_to_align, phase_factor


def masked_phase_for_plot(field, intensity=None, threshold=0.00):
    """
    Return phase with low-intensity pixels masked.

    Phase is not physically meaningful where the field amplitude is close to zero.
    """
    phase = np.angle(field)

    if intensity is None:
        intensity = np.abs(field) ** 2

    mask = intensity < threshold * np.max(intensity)

    return np.ma.array(phase, mask=mask)


def wrapped_phase_difference(phase_a, phase_b):
    """
    Wrapped phase difference in the range [-pi, pi].
    """
    return np.angle(np.exp(1j * (phase_a - phase_b)))


# --------------------------------------------------
# Step 4: Test fitting / reconstruct each generated field
# --------------------------------------------------
N_TEST = min(N_TEST, N_RANDOM_FIELDS)

intensity_rms_errors = []
complex_field_rms_errors = []
relative_complex_field_rms_errors = []
costs = []
nfevs = []
success_flags = []
global_phase_factors = []

example_target = None
example_fit = None
example_residual = None

example_target_field = None
example_fit_field = None
example_complex_residual_abs = None
example_target_phase = None
example_fit_phase = None
example_phase_residual = None

for i in range(N_TEST):
    print(f"\nFitting sample {i + 1}/{N_TEST}")

    target_image = intensities[i]
    target_field = fields_true_normalised[i]

    coeff_fit, intensity_fit, field_fit, res = fit_lp_coeffs_to_target_complex_field(
        mode_matrix_fit,
        target_field,
        max_nfev=MAX_NFEV,
        n_restarts=N_RESTARTS,
        rng=rng,
    )
    # Image/intensity RMS error
    intensity_rms = np.sqrt(np.mean((intensity_fit - target_image) ** 2))

    # --------------------------------------------------
    # Complex-field diagnostic comparison
    # --------------------------------------------------
    # The fit was found by matching the full complex field, so there is no
    # extra global phase alignment here. Any phase offset is part of the error.
    fit_field_aligned = field_fit
    global_phase_factor = 1.0 + 0.0j

    complex_residual = target_field - fit_field_aligned

    complex_field_rms = np.sqrt(np.mean(np.abs(complex_residual) ** 2))
    relative_complex_field_rms = (
        complex_field_rms
        / (np.sqrt(np.mean(np.abs(target_field) ** 2)) + 1e-12)
    )

    # Align coefficients before comparing them
   # coeff_fit_aligned = align_coeffs_to_true(coeff_fit, true_coeff)

   # coeff_rms = np.sqrt(np.mean(np.abs(coeff_fit_aligned - true_coeff) ** 2))

    #coeff_relative_rms = (
    #    np.linalg.norm(coeff_fit_aligned - true_coeff)
     #   / (np.linalg.norm(true_coeff) + 1e-12)
   # )

    intensity_rms_errors.append(intensity_rms)
    complex_field_rms_errors.append(complex_field_rms)
    relative_complex_field_rms_errors.append(relative_complex_field_rms)
    global_phase_factors.append(global_phase_factor)
    #coeff_rms_errors.append(coeff_rms)
    #coeff_relative_rms_errors.append(coeff_relative_rms)
    costs.append(res.cost)
    nfevs.append(res.nfev)
    success_flags.append(res.success)

    print("Intensity RMS:", intensity_rms)
    print("Complex field RMS:", complex_field_rms)
    print("Relative complex field RMS:", relative_complex_field_rms)
    #print("Coeff RMS:", coeff_rms)
    #print("Coeff relative RMS:", coeff_relative_rms)
    print("Optimiser success:", res.success)

    if i == EXAMPLE_INDEX:
        example_target = target_image
        example_fit = intensity_fit
        example_residual = intensity_fit - target_image

        example_target_field = target_field
        example_fit_field = fit_field_aligned
        example_complex_residual_abs = np.abs(complex_residual)

        example_target_phase = masked_phase_for_plot(
            example_target_field,
            intensity=example_target,
        )

        example_fit_phase = masked_phase_for_plot(
            example_fit_field,
            intensity=example_fit,
        )

        phase_residual = wrapped_phase_difference(
            np.angle(example_target_field),
            np.angle(example_fit_field),
        )

        phase_mask = (
            (example_target < 0.02 *
             np.max(example_target)) |
            (example_fit < 0.02 *
             np.max(example_fit))
        )

        example_phase_residual = phase_residual
        #np.ma.array(phase_residual, mask=phase_mask)


# --------------------------------------------------
# Step 5: Save test results
# --------------------------------------------------
intensity_rms_errors = np.array(intensity_rms_errors)
complex_field_rms_errors = np.array(complex_field_rms_errors)
relative_complex_field_rms_errors = np.array(relative_complex_field_rms_errors)
costs = np.array(costs)
nfevs = np.array(nfevs)
success_flags = np.array(success_flags)
global_phase_factors = np.array(global_phase_factors)

results_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_test_results_{N_TEST}tests_gen{N_MODES_GENERATE}_fit{N_MODES_FIT}modes.npz"
)

np.savez_compressed(
    results_save_path,
    intensity_rms_errors=intensity_rms_errors,
    complex_field_rms_errors=complex_field_rms_errors,
    relative_complex_field_rms_errors=relative_complex_field_rms_errors,
    costs=costs,
    nfevs=nfevs,
    success_flags=success_flags,
    global_phase_factors=global_phase_factors,
)

print("\nSaved fit results to:")
print(results_save_path)

# --------------------------------------------------
# Save CSV summary
# --------------------------------------------------
csv_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_test_summary_{N_TEST}tests_gen{N_MODES_GENERATE}_fit{N_MODES_FIT}modes.csv"
)

summary = np.column_stack([
    np.arange(N_TEST),
    intensity_rms_errors,
    complex_field_rms_errors,
    relative_complex_field_rms_errors,
    costs,
    nfevs,
    success_flags.astype(int),
])

np.savetxt(
    csv_save_path,
    summary,
    delimiter=",",
    header="sample,intensity_rms,complex_field_rms,relative_complex_field_rms,cost,nfev,success",
    comments="",
)

print("Saved CSV summary to:")
print(csv_save_path)
# --------------------------------------------------
# Print final averages
# --------------------------------------------------
print("\n==============================")
print("Overall complex-field fitting results")
print("==============================")
print("Mean intensity RMS:", np.mean(intensity_rms_errors))
print("Median intensity RMS:", np.median(intensity_rms_errors))
print("Mean complex field RMS:", np.mean(complex_field_rms_errors))
print("Median complex field RMS:", np.median(complex_field_rms_errors))
print("Mean relative complex field RMS:", np.mean(relative_complex_field_rms_errors))
print("Median relative complex field RMS:", np.median(relative_complex_field_rms_errors))
#print("Mean coeff RMS:", np.mean(coeff_rms_errors))
#print("Median coeff RMS:", np.median(coeff_rms_errors))
#print("Mean coeff relative RMS:", np.mean(coeff_relative_rms_errors))
#print("Median coeff relative RMS:", np.median(coeff_relative_rms_errors))
print("Success rate:", np.mean(success_flags))


# --------------------------------------------------
# Plot intensity diagnostics only
# --------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(example_target, cmap="inferno", vmin=0, vmax=1)
plt.title(r"Target intensity $|E_\mathrm{target}|^2$")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(example_fit, cmap="inferno", vmin=0, vmax=1)
plt.title(r"Fitted intensity $|E_\mathrm{fit}|^2$")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(example_residual, cmap="bwr")
plt.title(r"Intensity residual $I_\mathrm{fit} - I_\mathrm{target}$")
plt.colorbar()

plt.tight_layout()

intensity_plot_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_intensity_example_gen{N_MODES_GENERATE}_fit{N_MODES_FIT}modes.png"
)

plt.savefig(intensity_plot_save_path, dpi=300, bbox_inches="tight")

print("Saved intensity example plot to:")
print(intensity_plot_save_path)

plt.show()


# --------------------------------------------------
# Plot phase diagnostics only
# --------------------------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(example_target_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
plt.title(r"Target phase $\arg(E_\mathrm{target})$")
plt.colorbar(label="phase [rad]")

plt.subplot(1, 3, 2)
plt.imshow(example_fit_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
plt.title(r"Fit phase $\arg(E_\mathrm{fit})$")
plt.colorbar(label="phase [rad]")

plt.subplot(1, 3, 3)
plt.imshow(example_phase_residual, cmap="twilight", vmin=-np.pi, vmax=np.pi)
plt.title(r"Wrapped phase residual")
plt.colorbar(label="phase difference [rad]")

plt.tight_layout()

phase_plot_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_phase_example_gen{N_MODES_GENERATE}_fit{N_MODES_FIT}modes.png"
)

plt.savefig(phase_plot_save_path, dpi=300, bbox_inches="tight")

print("Saved phase example plot to:")
print(phase_plot_save_path)

plt.show()


# --------------------------------------------------
# Plot complex-field residual magnitude separately
# --------------------------------------------------
plt.figure(figsize=(5, 4))

plt.imshow(example_complex_residual_abs, cmap="inferno")
plt.title(r"Complex residual magnitude $|E_\mathrm{target} - E_\mathrm{fit}|$")
plt.colorbar()

plt.tight_layout()

complex_residual_plot_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_complex_residual_example_gen{N_MODES_GENERATE}_fit{N_MODES_FIT}modes.png"
)

plt.savefig(complex_residual_plot_save_path, dpi=300, bbox_inches="tight")

print("Saved complex residual magnitude plot to:")
print(complex_residual_plot_save_path)

plt.show()


# --------------------------------------------------
# Optional: save the exact complex arrays used in the example plot
# --------------------------------------------------
example_complex_save_path = os.path.join(
    outdir,
    f"LP_complex_field_fit_example_arrays_gen{N_MODES_GENERATE}_fit{N_MODES_FIT}modes.npz"
)

np.savez_compressed(
    example_complex_save_path,
    E_target_real=example_target_field.real,
    E_target_imag=example_target_field.imag,
    E_fit_real=example_fit_field.real,
    E_fit_imag=example_fit_field.imag,
    abs_Etarget_minus_Efit=example_complex_residual_abs,
    phase_target=np.angle(example_target_field),
    phase_fit=np.angle(example_fit_field),
    phase_residual=example_phase_residual,
)

print("Saved example complex arrays to:")
print(example_complex_save_path)
