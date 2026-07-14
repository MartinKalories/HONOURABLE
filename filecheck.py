import numpy as np

filename = "/home/manav/PL-NN-testdata_forDec2025/pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_files-combined-PSFs.npz"

with np.load(filename, allow_pickle=True) as npf:
    print(npf.files)
