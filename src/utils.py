import h5py
import numpy as np
from scipy.stats import zscore
from scipy.ndimage import binary_erosion
from spyeeg.models.TRF import TRFEstimator

def prepare_training_data(subject, train_indices, val_indices, test_indices, hdf5_path):
    """
    Returns data matrix, label etc. for regression model training
    Can be used for Cross-Validation with appropriate validaten and test indices

    X: EEG-data
    y: speech envelope

    Args:
        subject (string): subject identifier
        val_indices (np.array): indices for validation
        test_indices (np.array): indices for the test set
        dataset (string): 'complete' or 'single_speaker' whether to use the complete dataset or only the single speaker trials

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, y_competing
    """
    with h5py.File(hdf5_path, 'r') as f:
        trials = (list(f['eeg'][subject].keys()))
        if 'taken_out_indices' in trials:
            trials.remove('taken_out_indices')
        trials = sorted(trials, key = int)
        trials = np.array(trials)
        #print(f'trials: {trials}')
        
        train_parts = trials[train_indices]
        val_parts = trials[val_indices]
        test_parts = trials[test_indices]

        attended_stimuli = np.array([f[f'eeg/{subject}/{j}'].attrs['stimulus'] for j in trials])
        train_stimuli = attended_stimuli[train_indices]
        val_stimuli = attended_stimuli[val_indices]
        test_stimuli = attended_stimuli[test_indices]

        X_train = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in train_parts])
        X_val = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in val_parts])
        X_test = np.hstack([zscore(f[f'eeg/{subject}/{j}'][:], axis = 1) for j in test_parts])

        #drop aux channels
        X_train, X_val, X_test = X_train[:31,:], X_val[:31,:], X_test[:31,:]

        y_train = np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in train_stimuli])
        y_val =  np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in val_stimuli])
        y_attended = np.hstack([f[f'stimulus_files/{str(j)}/attended_env'][:] for j in test_stimuli])
        y_competing = np.hstack([f[f'stimulus_files/{str(j)}/distractor_env'][:] for j in test_stimuli])
    f.close()

    return X_train, X_val, X_test, y_train, y_val, y_attended, y_competing


def trf_helper(speech_feature, eeg, tmin, tmax, Fs, alpha, null_model = False, n_null = 500, get_scores = False):
    """
    Helper function to calculate TRF coefficients.

    Args:
        speech_feature (np.array): Speech feature
        eeg (np.array): EEG data
        tmin (float): Minimum time lag in seconds
        tmax (float): Maximum time lag in seconds
        Fs (int): Sampling frequency
        alpha (list): List of alpha values

    Returns:
        np.array: TRF coefficients
    """

    n_times = eeg.shape[-1]
    n_train = int(n_times * 0.8)
    speech_train, eeg_train = speech_feature[:n_train], eeg[:,:n_train]

    offsets = np.linspace(int(len(speech_train) / 3), int(len(speech_train) / 3 * 2), num=n_null, dtype=int).tolist()
    if null_model:
        coefs = []
        scores = []
        for offset in offsets:
            # roll the speech feature
            speech_train_null = np.roll(speech_train, offset)
            speech_test, eeg_test = speech_feature[n_train:], eeg[:, n_train:]
            trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha, alpha_feat=False)
            trf.fit(np.expand_dims(speech_train_null, axis=1), eeg_train.T)
            scores_trmp = trf.score(np.expand_dims(speech_test, axis=1), eeg_test.T)
            coefs_tmp = trf.get_coef()[:, 0, :, :].T
            coefs.append(coefs_tmp)
            scores.append(scores_tmp)
        coefs = np.array(coefs)
        scores = np.array(scores)
    else:
        speech_test, eeg_test = speech_feature[n_train:], eeg[:, n_train:]

        trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha, alpha_feat=False)
        trf.fit(np.expand_dims(speech_train, axis=1), eeg_train.T)

        scores = trf.score(np.expand_dims(speech_test, axis=1), eeg_test.T)

        mean_scores_alpha = np.mean(scores, axis=0)
        best_alpha = np.argmax(mean_scores_alpha)
        coefs = trf.get_coef()[:, 0, :, :].T
    
    if get_scores:
        return coefs, scores
    else:
        return coefs

def find_start_distractor(distr_arr, thres_window = 10):
    """
    Finds index where the distractor stars, by finding the first {thres_window} consecutive samples above the mean.

    Args:
        distr_arr (np.array): array of distractor envelope
        thres_window (int, optional): number of consecutive samples above the mean to be considered as start of distractor. Defaults to 10.

    Returns:
        int: index where the distractor starts
    """
    m = distr_arr > distr_arr.mean()
    k = np.ones(thres_window,dtype=bool)
    starting_index = binary_erosion(m,k,origin=-(thres_window//2)).argmax()
    return starting_index