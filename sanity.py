import numpy as np
import os

path = "/home/manav//PL-NN-testdata_forDec2025/slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"

with np.load(path, allow_pickle=True) as data:
    print("Keys in NPZ:")
    for key in data.keys():
        arr = data[key]
        print()
        print("key:", key)
        print("shape:", getattr(arr, "shape", None))
        print("dtype:", getattr(arr, "dtype", None))

        if np.issubdtype(arr.dtype, np.number):
            print("min:", np.nanmin(arr))
            print("max:", np.nanmax(arr))
            print("mean:", np.nanmean(arr))
