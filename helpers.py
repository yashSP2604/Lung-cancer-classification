import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import pickle
import os
from model import sigmoid


with open("model/model_params.pkl", 'rb') as f:
    data_wb = pickle.load(f)

with open("model/scaler.pkl", 'rb') as f:
    data_scaler = pickle.load(f)

w_final = data_wb["weights"]
b_final = data_wb["bias"]
scaler = data_scaler["scaler"]


def predict():
    user_Age = int(input("Enter value for Age: "))
    user_Smoking_Years = float(input("Enter value for Smoking_Years(0–50): "))
    user_Exposure_Asbestos = float(input("Enter value for Exposure_Asbestos(0 or 1): "))
    user_Family_History = float(input("Enter value for Family_History(0 or 1): "))
    user_Air_Quality_Index = float(input("Enter value for Air_Quality_Index(10 - 200): "))
    user_bmi = float(input("Enter value for bmi(15–40): "))
    user_exercise_frequency = float(input("Enter value for exercise_frequency(0 - 7): "))
    user_packs_per_day = float(input("Enter value for consuming smoke packs_per_day (0 - 3): "))

    user_array = np.array([user_Age,user_Smoking_Years,user_Exposure_Asbestos,user_Family_History,user_Air_Quality_Index,user_bmi,user_exercise_frequency,user_packs_per_day])
    user_array_scaled = scaler.transform(user_array.reshape(1,-1))

    z_user = np.dot(user_array_scaled,w_final)+b_final
    probability = sigmoid(z_user)
    predicted_class = 1 if probability >= 0.5 else 0
    
    print(f"Predicted probability: {probability*100}")
    print(f"Predicted class: {'High Risk(1)' if predicted_class == 1 else 'Low Risk(0)'}")


if __name__ == "__main__" :
    predict()


