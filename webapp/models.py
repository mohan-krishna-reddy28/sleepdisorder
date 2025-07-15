import os
import joblib
from tensorflow.keras.models import load_model # Make sure this import is correct for your Keras version

# Get the directory of the current file (models.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to your saved files
MODEL_PATH = os.path.join(CURRENT_DIR, 'cnn_model.h5')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler_n.pkl')
LABEL_ENCODERS_PATH = os.path.join(CURRENT_DIR, 'label_encoders.pkl') # Note the filename change!

# Load the model and preprocessors
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    print("Model and preprocessors loaded successfully!")
except Exception as e:
    print(f"Error loading model or preprocessors: {e}")
    # Handle the error appropriately, e.g., by raising it or using default values
    model = None
    scaler = None
    label_encoders = None

# You would then use 'model', 'scaler', and 'label_encoders' in your prediction logic
# (e.g., in a function or a view that handles predictions)
# Example (if placed in views.py or a prediction utility function):
# def predict_sleep_disorder(data):
#     # Apply label encoding to categorical features (using label_encoders dictionary)
#     for col, le in label_encoders.items():
#         if col in data: # Make sure the column exists in the input data
#             data[col] = le.transform([data[col]])[0] # Transform single value
#     # Apply scaling
#     scaled_data = scaler.transform([list(data.values())]) # Adjust this to match your feature order
#     # Make prediction
#     prediction = model.predict(scaled_data)
#     return prediction