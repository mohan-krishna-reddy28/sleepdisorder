from django.db import models
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("C:\\Users\\ADMIN\\Music\\FINANCIAL_FRAUD\\FRONTEND\\cnn_model.h5")

# Load scaler and encoders
scaler = joblib.load("C:\\Users\\ADMIN\\Music\\FINANCIAL_FRAUD\\FRONTEND\\scaler_n.pkl")
label_encoders = joblib.load("C:\\Users\\ADMIN\\Music\\FINANCIAL_FRAUD\\FRONTEND\\label_encoders (1).pkl")

# Prediction function
def predict(lst):
    # Convert input data to numpy array and reshape for CNN input
    test = np.array(lst).reshape(1, -1)
    
    # Scale the data
    test_scaled = scaler.transform(test)
    
    # Reshape for CNN input format
    test_reshaped = np.reshape(test_scaled, (test_scaled.shape[0], test_scaled.shape[1], 1))

    # Predict
    prediction = model.predict(test_reshaped)
    result = 'Fraudulent' if prediction[0][0] > 0.5 else 'Not Fraudulent'
    
    return result
