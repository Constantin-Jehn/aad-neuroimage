from spyeeg.models.TRF import TRFEstimator
import git
import numpy as np
import mne
import os
import matplotlib.pyplot as plt
import h5py
from mne_icalabel import label_components
import sys
sys.path.append("..")
from utils import trf_helper, find_start_distractor
from tqdm import tqdm


base_dir = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
hdf5_path = os.path.join(base_dir, 'data/nh_dataset_8Hz.hdf5')

target_dir = os.path.join(base_dir, 'results/trf')
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

subjects = ["303", "304"]

trials = list(range(9,21))

tmin = -0.3
tmax = 0.8
sfreq = 128
alpha = np.logspace(-7, 7, 15)
null_model = False

with h5py.File(hdf5_path, 'r') as f:
    coefs_cohort_attended = []
    coefs_cohort_distractor = []

    scores_cohort_attended = []
    scores_cohort_distractor = []

    for subject in tqdm(subjects):
        coefs_subj_attended = []
        coefs_subj_distractor = []
        
        scores_subj_attended = []
        scores_subj_distractor = []


        for trial in trials:
            eeg_path = f'eeg/{subject}/{trial}'
            stim_code = f[eeg_path].attrs['stimulus']
            env_attended_path = f'stimulus_files/{stim_code}/attended_env'

            env_attended = f[env_attended_path][:]
            eeg = f[eeg_path][:]
            eeg = eeg[:31,:]

            env_distractor_path =  f'stimulus_files/{stim_code}/distractor_env'
            env_distractor = f[env_distractor_path][:]

            # Cut data to where the distractor is present
            start_env_distractor = find_start_distractor(env_distractor)
            env_attended = env_attended[start_env_distractor:]
            env_distractor = env_distractor[start_env_distractor:]
            eeg = eeg[:,start_env_distractor:]

            coefs_attended, scores_attended = trf_helper(env_distractor, eeg, tmin=tmin, tmax=tmax, Fs=sfreq, alpha=alpha, null_model=null_model, get_scores=True)
            coefs_distractor, scores_distractor = trf_helper(env_distractor, eeg, tmin=tmin, tmax=tmax, Fs=sfreq, alpha=alpha, null_model=null_model, get_scores=True)

            coefs_subj_attended.append(coefs_attended)
            coefs_subj_distractor.append(coefs_distractor)

            scores_subj_attended.append(scores_attended)
            scores_subj_distractor.append(scores_distractor)
        
        coefs_cohort_attended.append(coefs_subj_attended)
        coefs_cohort_distractor.append(coefs_subj_distractor)

f.close()

# Save the coefficients and scores for further analysis
np.save(os.path.join(target_dir, 'trf_coefs_attended.npy'), coefs_cohort_attended)
np.save(os.path.join(target_dir, 'trf_coefs_distractor.npy'), coefs_cohort_distractor)

np.save(os.path.join(target_dir, 'trf_scores_attended.npy'), scores_cohort_attended)
np.save(os.path.join(target_dir, 'trf_scores_distractor.npy'), scores_cohort_distractor)


