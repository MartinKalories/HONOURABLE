"""
Find and display the guided LP modes of a step-index multimode fiber.

Optionally generates RSoft beamprop launch commands for each mode.
"""

from lanternfiber import lanternfiber
import numpy as np
import matplotlib

# # PyCharm's bundled pydev (pre-2024) doesn't recognise 'macosx' as a valid
# # IPython GUI name, even though newer matplotlib uses it for the MacOSX
# # backend. The block below silently maps it to the equivalent 'osx' entry.
# try:
#     from pydev_ipython.inputhook import guis as _pydev_guis
#     if 'macosx' not in _pydev_guis and 'osx' in _pydev_guis:
#         _pydev_guis['macosx'] = _pydev_guis['osx']
# except ImportError:
#     pass

import matplotlib.pyplot as plt


# Fiber parameters
n_core = 1.44
n_cladding = 1.4345
wavelength = 1.55   # microns
core_radius = 32.8/2  # microns

show_plots = True
sort_by_b = True  # if True, print modes sorted by propagation constant (descending)

f = lanternfiber(n_core, n_cladding, core_radius, wavelength)
f.find_fiber_modes()
f.make_fiber_modes(show_plots=show_plots, plot_pausetime=0.5)

print("Supported LP modes (l, m):")
if sort_by_b:
    b_lookup = {(f.allmodes_l[i], f.allmodes_m[i]): f.allmodes_b[i]
                for i in range(f.nLPmodes)}
    mode_list = sorted(f.lp_mode_list,
                       key=lambda mode: b_lookup[(abs(mode[0]), mode[1])],
                       reverse=True)
else:
    mode_list = f.lp_mode_list
for mode in mode_list:
    print("  LP%d%d" % (mode[0], mode[1]))

print("Total number of scalar modes:", f.nmodes)

# -------------------------------------------------------------------------
# Extracting mode fields as numpy arrays
#
# After make_fiber_modes(), the fields are stored as lists of 2D arrays
# indexed by LP mode group (not counting degenerate orientations).
# For modes with l > 0 there are two degenerate orientations (cos and sin).
#
#   f.allmodefields_cos_cart  — list of 2D arrays, cos-oriented, Cartesian
 #  f.allmodefields_sin_cart  — list of 2D arrays, sin-oriented, Cartesian
#   f.nLPmodes                — number of unique LP mode groups
#   f.nmodes                  — total modes including degenerate pairs
#   f.microns_per_pixel       — spatial scale of each array
# -------------------------------------------------------------------------

# Single mode field: LP01 (index 0), cos orientation
lp01_field = f.allmodefields_cos_cart[0]          # 2D numpy array, real amplitude
print("LP01 field shape:", lp01_field.shape)

# Physical coordinate axis (microns), centred on the fiber
npix_full = lp01_field.shape[0]
half_width_microns = f.microns_per_pixel * npix_full / 2
x = np.linspace(-half_width_microns, half_width_microns, npix_full)  # microns

# Stack all cos-oriented fields into a single 3D array: (nLPmodes, ny, nx)
all_cos_fields = np.array(f.allmodefields_cos_cart)
print("All cos mode fields shape:", all_cos_fields.shape)

# Stack all modes including degenerate sin partners: (nmodes, ny, nx)
# (same ordering used internally by the coupling calculations)
all_fields = np.array(f.allmodefields_rsoftorder)
print("All mode fields (with degeneracies) shape:", all_fields.shape)

#all_fields = np.array(f.allmodefields_rsoftorder)

# ------------------------------------------------------------
# Convert mode fields into a matrix
# ------------------------------------------------------------

# all_fields shape: (nmodes, ny, nx)
nmodes, ny, nx = all_fields.shape

# Flatten each 2D mode into a 1D vector
# Result shape: (ny*nx, nmodes)
mode_matrix = all_fields.reshape(nmodes, ny * nx).T

print("Mode matrix shape:", mode_matrix.shape)
# rows = pixels
# columns = modes


# ------------------------------------------------------------
# Example: select first mode only
# ------------------------------------------------------------

coeffs = np.zeros((nmodes, 1))
coeffs[0, 0] = 1.0

output_flat = mode_matrix @ coeffs      # shape: (ny*nx, 1)
output_field = output_flat.reshape(ny, nx)

print("Output field shape:", output_field.shape)


# Plot result
plt.figure()
plt.imshow(output_field, cmap="bwr")
plt.colorbar(label="Field amplitude")
plt.title("Reconstructed field from mode coefficients")
plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------
# Optional: generate RSoft beamprop launch commands for each mode.
# Set make_rsoft_commands = True and update indfile/base_prefix to match
# your RSoft project.
# -------------------------------------------------------------------------
#make_rsoft_commands = False

#if make_rsoft_commands:
    #indfile = 'your_structure.ind'
    #base_prefix = indfile[:-4] + '_scan01'
   # commands_outfile = './outputs/rsoft_commands.txt'
   # hide_sim_window = True

   # hide_cmd = '-hide ' if hide_sim_window else ''
   # all_cmds = []
   # for mode in f.lp_mode_list:
       # cur_prefix = base_prefix + '_LP%d%d' % (mode[0], mode[1])
       # cur_cmd = ('bsimw32 ' + hide_cmd + indfile +
        #           ' prefix=' + cur_prefix +
         #          ' launch_mode=%d' % mode[0] +
         #          ' launch_mode_radial=%d' % mode[1] +
         #          ' wait=0')
      #  all_cmds.append(cur_cmd)

 #   with open(commands_outfile, 'w') as fout:
    #    for cmd in all_cmds:
     #       fout.write(cmd + '\n')
  #  print("RSoft commands written to", commands_outfile)
