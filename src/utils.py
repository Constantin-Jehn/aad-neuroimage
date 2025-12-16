import h5py
import numpy as np
from scipy.stats import zscore
from scipy.ndimage import binary_erosion
from spyeeg.models.TRF import TRFEstimator
import mne

def preprocess_eeg_array(eeg, fs_in, fs_output, corner_freq_l, corner_freq_h):
    if corner_freq_h is not None:
        eeg = mne.filter.filter_data(eeg, fs_in, l_freq=corner_freq_l, h_freq=None)
    if fs_output != fs_in:
        down = fs_in/fs_output
        eeg = mne.filter.resample(eeg, down = down, axis=1)
    if corner_freq_l is not None:
        eeg = mne.filter.filter_data(eeg, sfreq = fs_output, l_freq = None, h_freq = corner_freq_h)
    return eeg

def concatenate_eeg(f, subject, trials, fs_in=1000, fs_output=128, corner_freq = [1,8]):
    eeg_data = []
    for j in trials:
        eeg_part = f[f'eeg/{subject}/{j}'][:]
        eeg_part = preprocess_eeg_array(eeg_part, fs_in, fs_output, corner_freq[0], corner_freq[1])
        eeg_data.append(zscore(eeg_part, axis = 1))
    eeg_data = np.hstack(eeg_data)
    return eeg_data

def concatenate_stimulus(f, stimuli, fs_in=1000, fs_output=128, feature_name='attended_env'):
    stim_data = []
    for j in stimuli:
        stim_part = f[f'stimulus_files/{str(j)}/{feature_name}'][:]
        if fs_output != fs_in:
            down = fs_in/fs_output
            stim_part = mne.filter.resample(stim_part, down = down)
        stim_data.append(zscore(stim_part))
    stim_data = np.hstack(stim_data)
    return stim_data

def prepare_training_data(subject, train_indices, val_indices, test_indices, hdf5_path, fs_in=1000, fs_output=128, corner_freq = [1,8]):
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

        X_train = concatenate_eeg(f, subject, train_parts, fs_in, fs_output, corner_freq)
        X_val = concatenate_eeg(f, subject, val_parts, fs_in, fs_output, corner_freq)
        X_test = concatenate_eeg(f, subject, test_parts, fs_in, fs_output, corner_freq)


        #drop aux channels
        X_train, X_val, X_test = X_train[:31,:], X_val[:31,:], X_test[:31,:]
        
        y_train = concatenate_stimulus(f, train_stimuli, fs_in, fs_output, feature_name='attended_env')
        y_val = concatenate_stimulus(f, val_stimuli, fs_in, fs_output, feature_name='attended_env')
        y_attended = concatenate_stimulus(f, test_stimuli, fs_in, fs_output, feature_name='attended_env')
        y_competing = concatenate_stimulus(f, test_stimuli, fs_in, fs_output, feature_name='distractor_env')

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
            trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
            trf.fit(np.expand_dims(speech_train_null, axis=1), eeg_train.T)
            scores_tmp = trf.score(np.expand_dims(speech_test, axis=1), eeg_test.T)
            coefs_tmp = trf.get_coef()[:, 0, :, :].T
            coefs.append(coefs_tmp)
            scores.append(scores_tmp)
        coefs = np.array(coefs)
        scores = np.array(scores)
    else:
        speech_test, eeg_test = speech_feature[n_train:], eeg[:, n_train:]

        dif_len = np.abs(eeg_test.shape[1] - speech_test.shape[0])
        if dif_len > 0:
            assert dif_len < 10, "Length difference too large"
            eeg_test, speech_test = crop_to_common_length(eeg_test, speech_test)

        trf = TRFEstimator(tmin=tmin, tmax=tmax, srate=Fs, alpha=alpha)
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

def crop_to_common_length(eeg, env):
    print(f'Cropping data to common length. EEG length: {eeg.shape[1]}, Env length: {env.shape[0]}')
    min_length = min(eeg.shape[1], env.shape[0])
    eeg_cropped = eeg[:, :min_length]
    env_cropped = env[:min_length]
    return eeg_cropped, env_cropped