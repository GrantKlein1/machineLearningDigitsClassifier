# Handwritten Digits Classifier

A machine learning project that classifies handwritten digits (0-9) using K-Nearest Neighbors algorithm.

## Project Overview

This Python program performs classification on the optical recognition of handwritten digits dataset. It uses the scikit-learn library to build a KNN classifier that can achieve high accuracy on digit recognition.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib

## Installation

1. Clone this repository
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv sklearn-env
3. Activate the virtual environment:
   Windows: sklearn-env\Scripts\activate
   Linux/Mac: source sklearn-env/bin/activate

## Usage
Run the program from the command line in the project folder:
   python digitsTest.py

Workflow:
1. **Choose to train or load a model:**
     Option (a): Train a new model
     Option (b): Load a previously trained saved model
2. **Provide test data:**
     The program will prompt for a test data file
3. **View results:**
     The program outputs each test instance with its predicted classification and overall accuracy

## Dataset
This project uses the Optical Recognition of Handwritten Digits dataset as outlined in optdigits.names
- optdigits.tra: Training data (64 features per instance)
- optdigits.tes: Test data
- optdigits.names: Dataset description
- 10 classes (digits 0-9)
- 64 features representing 8x8 image of a digit

## Model Details
- **Algorithm:** K-Nearest Neighbors (KNN)
- **k value:** 1
- **Expected Accuracy:** ~98% on test set (similar accuracy was found for k values 1-10)
