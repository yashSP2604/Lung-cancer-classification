import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

# Utility Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        f_wb_i = sigmoid(np.dot(w, x[i]) + b)
        err_i = f_wb_i - y[i]
        dj_dw += err_i * x[i]
        dj_db += err_i
    return dj_dw / m, dj_db / m

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, verbose=False):
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        if verbose and (i % 500 == 0 or i == num_iters - 1):
            print(f'Iteration {i+1}/{num_iters}')
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

def predict(x, w, b):
    z = np.dot(x, w) + b
    return (sigmoid(z) >= 0.5).astype(int)

# Data Preparation
df = pd.read_csv('synthetic_lung_cancer_data_with_packs.csv')
features = ["age", "smoking_years", "asbestos_exposure", "family_history",
           "air_pollution_index", "bmi", "exercise_frequency", "packs_per_day"]
x = df[features].values
y = df["lung_cancer"].values

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model Training
w_in = np.zeros(x_train_scaled.shape[1])
b_in = 0.
alpha = 1.0e-2
num_iters = 5000
w_final, b_final = gradient_descent(x_train_scaled, y_train, w_in, b_in, alpha, num_iters, verbose=True)

# Training/Test Evaluation
train_preds = predict(x_train_scaled, w_final, b_final)
test_preds = predict(x_test_scaled, w_final, b_final)
acc_train = accuracy_score(y_train, train_preds)
acc_test = accuracy_score(y_test, test_preds)
print(f"\nTraining Accuracy: {acc_train:.2f}")
print(f"Test Accuracy: {acc_test:.2f}")


# ROC Curve (Test)
prob_test = sigmoid(np.dot(x_test_scaled, w_final) + b_final)
fpr_test, tpr_test, _ = roc_curve(y_test, prob_test)
auc_test = auc(fpr_test, tpr_test)
plt.figure()
plt.plot(fpr_test, tpr_test, label=f"Test ROC curve (AUC = {auc_test:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.title("Test ROC Curve for Lung Cancer Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# User Prediction (Example)
print("------------------------------------------------------------------")
user_input = [
    float(input("Enter value for Age: ")),
    float(input("Enter value for Smoking_Years(0–50): ")),
    float(input("Enter value for Exposure_Asbestos(0 or 1): ")),
    float(input("Enter value for Family_History(0 or 1): ")),
    float(input("Enter value for Air_Quality_Index(10 - 200): ")),
    float(input("Enter value for bmi(15–40): ")),
    float(input("Enter value for exercise_frequency(0 - 7): ")),
    float(input("Enter value for packs_per_day (0 - 3): "))
]
print("------------------------------------------------------------------")
user_scaled = scaler.transform([user_input])
user_pred = predict(user_scaled, w_final, b_final)


print("******************************************************************")
# Probability (between 0 and 1)
user_prob = sigmoid(np.dot(user_scaled, w_final) + b_final)[0]

# Convert to percentage
risk_percent = user_prob * 100

print(f"Estimated lung cancer risk: {risk_percent:.2f}%")

if user_prob >= 0.5:
    print("Model prediction: High lung cancer risk")
else:
    print("Model prediction: Low lung cancer risk")

print("******************************************************************\n")