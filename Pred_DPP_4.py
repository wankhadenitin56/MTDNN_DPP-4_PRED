
"""
MTDNN_DPP-4 is an advanced Deep Neural Network tool designed for predicting DPP-4 inhibitors.
Using the powerful PaDEL descriptor calculations,
this tool leverages deep learning to identify inhibitors and provide precise regression scores for DPP-4.
Ensure that compounds' SMILES are stored in a .smi file within the same directory.
Additionally, both the trained model (multitasking_model.h5) 
and the PaDEL folder must reside in this directory.
Following execution, the prediction results, including classification, probability, and regression scores, are conveniently saved in DPP-4-multitasking_predictions.csv.
With these insights, researchers can accelerate drug discovery processes confidently.

Edited by:
    Nitin Wankhade 
"""
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pickle

# Set your default path here
default_path = "E:/..../Prediction"
path = "E:/..../Prediction"

# Create function to get predictions using trained model
def dpp4_multitasking(input_dpp4: pd.DataFrame, scaler: StandardScaler, loaded_model) -> tuple:
    # Transform user data to numpy to avoid conflict with names
    dpp4_user_input = scaler.transform(input_dpp4.to_numpy())

    # Get predictions for user input
    predictions = loaded_model.predict(dpp4_user_input)

    # Classification prediction
    class_prediction = predictions[0]

    # Regression prediction
    regression_prediction = predictions[1][:, 0]

    # For functional models, calculate class probabilities 
    class_probabilities = np.zeros((len(dpp4_user_input), 2))
    class_probabilities[:, 1] = class_prediction  # Probability of positive class
    class_probabilities[:, 0] = 1 - class_prediction  # Probability of negative class

    return class_prediction, regression_prediction, class_probabilities

# Create main function to run descriptor calculation and predictions
def run_multitasking_prediction(folder: str) -> None:
    # Update the paths for PaDEL-Descriptor and descriptors.xml
    padel_cmd = [
        'java', '-jar',
        os.path.join(path, 'PaDEL-Descriptor/PaDEL-Descriptor.jar'),
        '-descriptortypes',
        os.path.join(path, 'PaDEL-Descriptor/descriptors.xml'),
        '-dir', folder, '-file', folder + '/PaDEL_features.csv',
        '-2d', '-fingerprints', '-removesalt', '-detectaromaticity',
        '-standardizenitro'
    ]

    # Calculate features
    subprocess.call(padel_cmd)
    print("Features calculated")

    # Create DataFrame for calculated features
    input_dpp4 = pd.read_csv(folder + "/PaDEL_features.csv")

    print("Number of features in input data:", input_dpp4.shape[1])

    # Replace "infinity" and very large values with zeros
    input_dpp4 = input_dpp4.replace([np.inf, -np.inf], np.nan)
    input_dpp4 = input_dpp4.fillna(0)  

    # Check column names and types in input data
    print("Input Data Columns:")
    print(input_dpp4.columns)
    print("Input Data Types:")
    print(input_dpp4.dtypes)

    # Keep only the numeric features present in the training data
    numeric_columns = input_dpp4.select_dtypes(include=['number']).columns
    input_dpp4 = input_dpp4[numeric_columns]

    # Load the scaler from the pickle file
    with open('scaler_params.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load multitasking model
    loaded_model = load_model(os.path.join(path, "multitasking_model.h5"))
    print("Model loaded")

    # Run multitasking predictions
    class_pred, regression_pred, class_probabilities = dpp4_multitasking(input_dpp4, scaler, loaded_model)
    print("Classification result: ", class_pred)
    print("Regression result (pIC50): ", regression_pred)

    # Convert regression output (pIC50) to IC50 nanomolar
    ic50_nanomolar = 10 ** (-regression_pred) * 10**9

    # Create DataFrame with results
    res = pd.DataFrame(index=input_dpp4.index)

    # Apply threshold for classification
    threshold = 0.95  # Change this threshold as per your requirements
    res['Predicted_class'] = (class_pred > threshold).astype(int)

    # Interpret class predictions as inhibitor (1) and non-inhibitor (0)
    res['Predicted_class'] = res['Predicted_class'].map({0: 'Non-Inhibitor', 1: 'Inhibitor'})

    # Save classification probabilities
    res['Probability'] = class_probabilities[:, 1]  # class 1 is the positive class

    # Save both the pIC50 in Molar and IC50 in Nanomolar
    res['Regression_output pIC50 in Molar'] = regression_pred
    res['Regression_output IC50 in Nanomolar'] = ic50_nanomolar

    # Print the results
    print("Predicted Class:", res['Predicted_class'].values)
    print("Probability:", res['Probability'].values)
    print("Raw Regression Output (pIC50):", res['Regression_output pIC50 in Molar'].values)
    print("Converted Regression Output (IC50 nanomolar):", res['Regression_output IC50 in Nanomolar'].values)

    # Save results to csv
    res.to_csv("E:/..../Prediction/DPP-4-multitasking_predictions.csv", index=False)

    return None

# Get multitasking predictions
run_multitasking_prediction(os.getcwd())
