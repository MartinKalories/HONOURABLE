# Photonic Lantern Mode Prediction

Deep learning framework for predicting optical wavefronts and point-spread functions from photonic lantern images using neural networks and Bayesian optimization.

## Overview

This project develops and optimizes convolutional neural networks to reconstruct optical wavefront information from photonic lantern (PL) images—a key capability for astronomical adaptive optics and precision optical metrology. The toolkit includes:

- **Fiber mode simulation** based on step-index fiber theory
- **Multi-output CNN models** for simultaneous WF and PSF prediction
- **Comprehensive Bayesian hyperparameter optimization**
- **Analysis tools** for parameter sensitivity and optimization landscape visualization

## Project Structure

```
├── Core Libraries
│   ├── lanternfiber.py           # Fiber optics mode calculations (LP mode solving)
│   ├── viewRSoftData.py          # RSoft optical simulation result parsing
│   ├── model_code.py             # Base NN architecture and training utilities
│
├── Training Modules
│   ├── firstopt.py               # Initial training configuration
│   ├── secondopt.py              # Enhanced training (30 epochs)
│   ├── thirdopt.py               # Production training (50 epochs)
│   ├── OPTIMUSV1.py              # Optimized full training (100 epochs)
│   ├── Optimusfixv1.py           # Quick optimization variant
│   └── stresstest.py             # Minimal stress test
│
├── Bayesian Optimization
│   ├── minimised_opt.py          # Minimal (5 parameters)
│   ├── leasil.py                 # Reduced space (4 parameters)
│   ├── miniopt_wgraph.py         # With live plotting
│   ├── minoptwgraphog.py         # Original variant
│   ├── clearmind.py              # Full optimization (14 parameters)
│   ├── SAVEMEMP4.py              # Resumable with checkpointing
│   └── optimise_model.py         # Extended parameter search
│
├── Analysis & Visualization
│   ├── cornerplot.py             # Scatter matrix from optimization results
│   ├── FAFO_ONG.py               # KDE-enhanced corner plot
│   ├── THEKDFILES.py             # Parameter dependency analysis
│   ├── trialkde.py               # Simple KDE example
│   └── TVAR.py                   # Temperature-varied KDE analysis
│
└── Experiments
    ├── example_fiber_modes.py    # Educational LP mode example
    ├── Bdawgbasewedits.py        # Main training pipeline
    ├── weretryingshit.py         # Synthetic mode-based data generation
    └── itsallinthere.py          # Mode-to-image decomposition
```

## Installation

### Requirements
- Python 3.7+
- TensorFlow 2.x
- scikit-optimize
- NumPy, SciPy, Pandas
- Matplotlib
- scikit-learn

### Setup

```bash
# Clone repository
git clone <repository>
cd HONOURABLE

# Install dependencies
pip install tensorflow scikit-optimize numpy scipy pandas matplotlib scikit-learn

# (Optional) For visualizations
pip install corner
```

## Usage

### 1. **Generate Fiber Modes**

Display guided LP modes for a standard step-index fiber:

```bash
python example_fiber_modes.py
```

Outputs mode field arrays and visualizations.

### 2. **Train a Model**

Run the production training pipeline:

```bash
python Bdawgbasewedits.py
```

Configure dataset paths and hyperparameters in the script header:
- `datadir`: Path to PL image and WF/PSF datasets
- `use_subset`: Number of samples to use (or `None` for all)
- `pdict`: Model hyperparameters (activation, learning rate, dropouts, layer sizes, etc.)

### 3. **Optimize Hyperparameters**

Run Bayesian optimization to find best hyperparameters:

```bash
python clearmind.py        # Full 14-parameter optimization
python minimised_opt.py     # Minimal 5-parameter optimization
python SAVEMEMP4.py         # Resumable optimization with checkpointing
```

Outputs:
- CSV file with all trial parameters and loss values
- Best parameters in JSON format
- Live loss curves

### 4. **Analyze Optimization Results**

Visualize the optimization landscape:

```bash
# Generate corner plot
python cornerplot.py --csv bayesopt_<date>_all_trials.csv --figscale 3.0

# KDE analysis
python FAFO_ONG.py --csv bayesopt_<date>_all_trials.csv --top-n 5

# Parameter dependency analysis
python THEKDFILES.py
```

## Key Modules

### `lanternfiber.py`
- **Class:** `lanternfiber(n_core, n_cladding, core_radius, wavelength)`
- **Methods:**
  - `find_fiber_modes()`: Solve eigenvalue equation for guided LP modes
  - `make_fiber_modes(show_plots=True)`: Generate normalized field distributions
  - `allmodefields_rsoftorder`: Get all mode fields as 3D array `(nmodes, ny, nx)`
- **Usage:**
  ```python
  f = lanternfiber(n_core=1.44, n_cladding=1.4345, core_radius=16.4, wavelength=1.55)
  f.find_fiber_modes()
  f.make_fiber_modes(show_plots=False)
  all_fields = np.array(f.allmodefields_rsoftorder)  # (nmodes, ny, nx)
  ```

### `model_code.py` / Training Modules
- **Function:** `train_one_run(pdict_override=None, do_predictions=True, do_plotting=True, ...)`
- **Inputs:**
  - `pdict_override`: Dict with hyperparameters to override defaults
  - Configuration flags for outputs (predictions, plots, models, movies)
- **Outputs:**
  - Training history (val/train losses)
  - Best predictions on test set
  - Trained model files
- **Default Model Architecture:**
  - Encoder: Conv2D layers → Flatten → Dense bottleneck
  - Decoder heads: WF branch (UpSampling2D path) and PSF branch (Dense path)
  - Dropout for regularization at multiple levels

## Data Format

### Input Data
- **PL images:** Shape `(n_samples, 128, 128)` normalized to zero-mean, unit-variance
- **WF images:** Shape `(n_samples, ny, nx)` wavefront in radians
- **PSF images:** Shape `(n_samples, 48, 48)` point-spread function intensity

### Precombined Dataset Files
Datasets are stored as `.npz` files with keys:
- `all_plims`: Input PL images
- `all_wfims`: Target wavefront images
- `all_psfims`: Target PSF images

Load example:
```python
data = np.load('pllabdata_20240605_combined.npz', allow_pickle=True)
all_plims = data['all_plims']
```

## Model Hyperparameters

Key tunable parameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `learningRate` | 1e-5 to 5e-3 | Adam optimizer learning rate (log-uniform) |
| `dropout_rate` | 0.0 to 0.4 | Input/intermediate dropout |
| `dropout_rate_dense` | 0.0 to 0.6 | Dense layer dropout |
| `dropout_rate_psf` | 0.0 to 0.8 | PSF branch dropout |
| `n_units_dense` | 512 to 4096 | Bottleneck dense layer size |
| `batchSize` | [16, 32, 64] | Training batch size |
| `ksz_enc` | [3, 5, 7] | Encoder kernel size |
| `nfilts_enc` | [64, 96, 128] | Encoder filters |
| `loss_weight` | 0.5 to 3.0 | PSF loss weight relative to WF |
| `epochs` | Variable | Training epochs |
| `actFunc` | [relu, elu, gelu] | Activation function |

## Optimization Strategy

The project uses **Gaussian Process-based Bayesian optimization** (scikit-optimize) to find optimal hyperparameters:

```python
from skopt import gp_minimize

result = gp_minimize(
    func=objective,           # Training function returning validation loss
    dimensions=space,         # Parameter search space
    n_calls=50,              # Total evaluations
    n_initial_points=10,     # Random initialization
    acq_func='EI'            # Expected Improvement acquisition
)
```

Best results are typically found after **50-100 trials**, with fastest improvements in first 20 trials.

## Visualization Tools

### Corner Plots
Scatter matrix showing 2D projections of all parameters colored by validation loss:

```bash
python cornerplot.py --csv bayesopt_20260419_all_trials.csv \
    --cols learningRate dropout_rate n_units_dense \
    --log-cols learningRate n_units_dense \
    --top-n 5 --figscale 3.0
```

### KDE Analysis
Smooth density estimation with top-trial highlighting:

```bash
python FAFO_ONG.py --csv <results.csv> --top-n 3 --levels 8
```

## Example Workflow

### 1. Generate and Visualize Modes
```python
from lanternfiber import lanternfiber
import numpy as np

f = lanternfiber(1.44, 1.4345, 16.4, 1.55)
f.find_fiber_modes()
f.make_fiber_modes(show_plots=True)
print(f"Total modes: {f.nmodes}")
```

### 2. Train with Custom Parameters
```bash
# Edit Bdawgbasewedits.py to set data paths and parameters
python Bdawgbasewedits.py
```

### 3. Optimize Hyperparameters
```bash
python clearmind.py  # Runs 50+ trials, saves results
```

### 4. Analyze Results
```bash
python cornerplot.py --csv bayesopt_*.csv --top-n 10
python FAFO_ONG.py --csv bayesopt_*.csv
```

## Performance Notes

- **Training time:** ~5-15 min per trial (depends on dataset size, GPU availability)
- **Typical best validation loss:** 0.0X (MSE on normalized images)
- **Optimization convergence:** Diminishing returns after ~40 trials
- **Hardware:** GPU recommended (CUDA/cuDNN); CPU training very slow

## Citation

If you use this code, please cite:

```
[Project details to be added]
```

## License

[Add license information]

## Contact

[Add contact info]