import sys
import git
import os
sys.path.append("..")
from ridge import Ridge
from utils import prepare_training_data
import numpy as np


##### Set data paths #####
hdf5_path = '/data_nfs/do00noto/semeco_data/data/processed/nh_dataset_1kHz.hdf5'

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

subjects = ["301"]


#### Set data parameters #####
fs_in = 1000 # Hz of dataset
fs_computation = 64 # Hz for model training and evaluation to speed up computation
corner_freq = [1,8] # Hz


##### Define hyperparameters #####
start_ms = -500
end_ms = 500
start_lag = int(start_ms/1000*fs_computation)
end_lag = int(end_ms/1000*fs_computation)
alpha = np.logspace(-7, 7, 15)

#### Define training, validation and test trials #####
test_indices = np.array([9])
val_indices = np.array([10])
train_indices = np.array([11,12])

#For full data training
#train_indices = np.delete(np.linspace(0,19,20, dtype=int), np.hstack((test_indices, val_indices)))




if __name__ == "__main__":
    ### load data ###
    X_train, X_val, X_test, y_train, y_val, y_attended, y_competing = prepare_training_data(
        subject=subjects[0],
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        hdf5_path=hdf5_path,
        fs_in=fs_in,
        fs_output=fs_computation,
        corner_freq=corner_freq
    )
    #### Initialize the model and train ####
    mdl = Ridge(start_lag=start_lag, end_lag=end_lag, alpha=alpha)
    mdl.fit(X_train.T, y_train[:,np.newaxis])

    # Choose best alpha on validation set
    _ = mdl.model_selection(X_val.T, y_val[:,np.newaxis])

    #### Evaluate on test set ###
    rec_scores_attended = mdl.score(X_test.T, y_attended[:,np.newaxis], best_alpha=True)
    rec_scores_competing = mdl.score(X_test.T, y_competing[:,np.newaxis], best_alpha=True)

    print(f"Reconstruction score attended: {rec_scores_attended}")
    print(f"Reconstruction score competing: {rec_scores_competing}")




