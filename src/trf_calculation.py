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
from utils import trf_helper, find_start_distractor, preprocess_eeg_array
from tqdm import tqdm


base_dir = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir
hdf5_path = os.path.join(base_dir, 'data/nh_dataset_8Hz.hdf5')

hdf5_path = '/data_nfs/do00noto/semeco_data/data/processed/nh_dataset_1kHz.hdf5'

target_dir = os.path.join(base_dir, 'results/trf')
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


"""
For more entire data analysis, use the following subject lists:
### NH Subjects:
subjects_nh = [
    301, 302, 303, 304, 305, 306, 307, 308, 309, 311, 313, 314, 315, 316, 317, 318, 319,
    321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 333
]
subjects = [str(s) for s in subjects_nh]

#### HI Subjects:
hi_subjects = [
    201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
    218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229
]
subjects_hi = [str(s) for s in hi_subjects]

#### CI Subjects:
ci_subjects = [
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 118, 119, 120,
    121, 122, 123, 124, 125, 127, 128, 130
]
subjects_ci = [str(s) for s in ci_subjects]
"""

subjects = ["313"]


trials = list(range(9,21))

tmin = -0.3
tmax = 0.8
sfreq_in = 1000
sfreq = 64

downsample_factor = sfreq_in/sfreq

corner_freq = [1,8] # Hz

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
            # Preprocess EEG
            

            env_distractor_path =  f'stimulus_files/{stim_code}/distractor_env'
            env_distractor = f[env_distractor_path][:]

            # Cut data to where the distractor is present
            start_env_distractor = find_start_distractor(env_distractor)
            env_attended = env_attended[start_env_distractor:]
            env_distractor = env_distractor[start_env_distractor:]

            # Preprocess and downsample EEG and stimulus
            eeg = eeg[:,start_env_distractor:]

            # envelope is already filtered in dataset
            env_attended = mne.filter.resample(env_attended, down = downsample_factor)
            env_distractor = mne.filter.resample(env_distractor, down = downsample_factor)
            eeg = preprocess_eeg_array(eeg, sfreq_in, sfreq, corner_freq[0], corner_freq[1])

            # TRF estimation
            coefs_attended, scores_attended = trf_helper(env_distractor, eeg, tmin=tmin, tmax=tmax, Fs=sfreq, alpha=alpha, null_model=null_model, get_scores=True)
            coefs_distractor, scores_distractor = trf_helper(env_distractor, eeg, tmin=tmin, tmax=tmax, Fs=sfreq, alpha=alpha, null_model=null_model, get_scores=True)

            # Store coefficients and scores
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


