# Repository for EEG analysis

This repository accompanies the paper "Attention Decoding at the Cocktail Party: Preserved in Hearing  Aid Users, Reduced in Cochlear Implant Users"

A pre-print can be found at: 

## Getting started
- Required packages are listed in the requirments.txt file
- sPyEEG - the tool used for TRF analysis can either be installes from source using python setup.py install (or pip -e) using the included folder or directly from pip for more info see https://github.com/phg17/sPyEEG

## Data

This repository is intended to provide analysis scripts for selective attention in cocktail-party situations of Cochlear-Implant users, hearing aid users and a typical hearing control group.
The corresponding data is available on zenodo:
- Cochlear Implant data: 
- Hearing Aided data: 10.5281/zenodo.17927767
- Typical Hearing data: 10.5281/zenodo.17952231
- Cochlear Implant data: 10.5281/zenodo.17952844


## Structure
- The data folder is intended to store the datasets to. The datasets can be downloaded from zenodo - links
- sPyEEG includes the source Code of the sPYEEG package
- src includes:
  - ridge.py: the linear backward model
  - train_eval_ridge.py: trains and evaluates the backward model
  - trf_calculation.py: trains the forward model
  - utils.py: contains some helper function for data handling

## Contact
The Code was written by Constantin Jehn (constantin.jehn@fau.de) at https://www.neurotech.tf.fau.eu

