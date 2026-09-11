import numpy as np

datadir = "/home/manav/PL-NN-testdata_forDec2025/"

# Dataset 1
pl_file_1 = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"
psf_file_1 = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined-PSFs.npz"
wf_file_1 = "slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"

# Dataset 2
pl_file_2 = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_proc2_files-combined.npz"
psf_file_2 = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_proc2_files-combined-PSFs.npz"
wf_file_2 = "slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_files-64px_combined.npz"

# LOAD DATASET 1


pl1 = np.load(datadir + pl_file_1, allow_pickle=True)
psf1 = np.load(datadir + psf_file_1, allow_pickle=True)
wf1 = np.load(datadir + wf_file_1, allow_pickle=True)

all_plims_1 = pl1["all_plims"]
all_filenames_1 = pl1["all_slmims_filenames"]

all_psfims_1 = psf1["all_psfims"]

all_pupphase_1 = wf1["all_pupphase"]



# LOAD DATASET 2


pl2 = np.load(datadir + pl_file_2, allow_pickle=True)
psf2 = np.load(datadir + psf_file_2, allow_pickle=True)
wf2 = np.load(datadir + wf_file_2, allow_pickle=True)

all_plims_2 = pl2["all_plims"]
all_filenames_2 = pl2["all_slmims_filenames"]

all_psfims_2 = psf2["all_psfims"]

all_pupphase_2 = wf2["all_pupphase"]

# CONCATENATE


all_plims = np.concatenate((all_plims_1, all_plims_2),axis=0)

all_filenames = np.concatenate((all_filenames_1, all_filenames_2),axis=0)

all_psfims = np.concatenate((all_psfims_1, all_psfims_2),axis=0)

all_pupphase = np.concatenate((all_pupphase_1, all_pupphase_2), axis=0)

# SHUFFLE
rng = np.random.default_rng(42)

shuffle_indices = rng.permutation(len(all_plims))
all_plims = all_plims[shuffle_indices]
all_filenames = all_filenames[shuffle_indices]
all_psfims = all_psfims[shuffle_indices]
all_pupphase = all_pupphase[shuffle_indices]

# SAVE
np.savez(datadir + "combined_two_datasets_PL.npz", all_plims=all_plims, all_slmims_filenames=all_filenames)

np.savez(datadir + "combined_two_datasets_PSF.npz", all_psfims=all_psfims)

np.savez(datadir + "combined_two_datasets_WF.npz",all_pupphase=all_pupphase, slmloc=wf1["slmloc"])

print("Saved combined datasets.")
