# MTDNN_DPP-4_PREd
**MTDNN_DPP-4_PREd** is a Deep Neural network-based model for prediction of DPP-4 inhibitors using 
Simplified Molecular Input Line Entry System (SMILES) notation of Compounds.

## Contents

The files contained in this repository are as follows:
 * ``prediction_script.py``: Main script to run predictions
 * ``smiles.smi``: User input structures (multiple)
 * ``multitasking_model.h5``: MTDNN prediction model
 * ``scaler_params.pkl``: Training dataset
 * ``PaDEL``: Folder with executable for feature calculation

## Requirements

* Python =(Version=3.9)
* Numpy=(Version=1.22)
* Pandas=(Version=1.5)

## Usage

In order to run DPP-4 inhibitor predictions, save input structures as SMILES in a single 
file (e.g. ``smiles.smi``) or input SMILES notation. 
 
1. Download this repository and ensure that all the files are present in the same folder when running the script.
2. Run ``prediction_script.py``. 
  ```bash
  python prediction_script.py <folder>
  ```
   If ``<folder>`` is not provided, the script runs in the current directory.
   A csv file (``...csv``) will be created in the folder where the script is run.
   A file containing the features generated by PaDEL will be also saved to disk (``PaDEL_features.csv``).
  
> **_NOTE:_** Remember to activate the corresponding conda environment before running the script, if applicable.
3. Prediction results will be saved in ``DPP-4-multitasking_predictions.csv`` which includes predicted class,regression score and associated probability

