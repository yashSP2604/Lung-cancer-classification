

# Lung Cancer Risk Classification

This project builds a logistic regression model on a synthetic lifestyle and exposure dataset to estimate the probability of developing lung cancer and classify individuals into low or high risk groups.

## Project overview
### Goal: 
Predict individual lung cancer risk based on factors such as age, smoking history, asbestos exposure, family history, air quality index, body mass index (BMI), exercise frequency, and packs smoked per day.

### Approach: 
Train a logistic regression classifier on a labeled synthetic dataset using a train/test split and evaluate model performance using accuracy and ROC–AUC.

### Key results: 
The final model achieves around 0.97 training accuracy and 0.99 test accuracy, with a test ROC–AUC close to 1.0, indicating excellent separation between low- and high-risk classes.

## Dataset
Type: Synthetic tabular dataset generated to mimic realistic distributions of lung cancer risk factors.

## Features:

Age (years)

Smoking years (0–50)

Exposure to asbestos (0 or 1)

Family history of lung cancer (0 or 1)

Air quality index (10–200)

BMI (15–40)

Exercise frequency (0–7 days per week)

Packs per day (0–3)

Target: Binary label indicating low vs. high lung cancer risk (and a corresponding probability output from the logistic regression model).

Split: The dataset is randomly split into training and test sets (for example, 80% train / 20% test) to evaluate generalization.

Since the data are synthetic, no real patient information is used, which avoids privacy issues and makes the project suitable for educational and experimental purposes.

## Methodology
Model:

Logistic regression classifier implemented with standard machine learning libraries (e.g., scikit-learn).

Preprocessing:

Train/test split.

feature scaling/normalization.

Training:

Optimization over multiple iterations (e.g., 5 000) to minimize the logistic loss and learn feature weights.

Training logs show convergence and report final training and test accuracy.

Evaluation:

Accuracy on train and test sets.

ROC curve and Area Under the Curve (AUC) to assess discrimination; the plotted ROC curve shows an AUC of approximately 1.00 for the test set.

## Results Performance:

Training accuracy: 0.97.

Test accuracy: 0.99.

Test ROC–AUC: ≈ 1.00, indicating the model almost perfectly distinguishes low-risk from high-risk individuals on the held-out test data.

## Example predictions:

A young, non-smoking individual with good air quality, healthy BMI, and regular exercise receives an estimated risk of around 2–3% and is classified as low lung cancer risk.

An older individual with many years of smoking, asbestos exposure, family history, higher air pollution, and lower exercise receives an estimated risk close to 90% and is classified as high lung cancer risk.
