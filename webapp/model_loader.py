# FRONTEND/webapp/model_loader.py
import os
import joblib
import tensorflow as tf
import logging # Added logging for better output

# Configure logging for model_loader
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# BASE_DIR is the directory of this file: .../FRONTEND/webapp/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Since your model files (scaler.pkl, CNN_Sleep_quantized.tflite, etc.)
# are located directly in the 'FRONTEND' folder (one level up from 'webapp'),
# we need to adjust the MODELS_DIR path accordingly.
MODELS_DIR = os.path.dirname(BASE_DIR) # <-- THIS IS THE CRITICAL CHANGE

logger.info(f"Attempting to load models from: {MODELS_DIR}") # Added for debugging confirmation

# --- Load the StandardScaler ---
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
try:
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Scaler loaded from: {SCALER_PATH}")
except Exception as e:
    logger.error(f"Error loading scaler from {SCALER_PATH}: {e}")
    scaler = None # Set to None if loading fails

# --- Load the Label Encoders ---
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, 'label_encoders.pkl')
try:
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    logger.info(f"Label encoders loaded from: {LABEL_ENCODERS_PATH}")
except Exception as e:
    logger.error(f"Error loading label encoders from {LABEL_ENCODERS_PATH}: {e}")
    label_encoders = None

# --- Load the Random Forest Model ---
RF_MODEL_PATH = os.path.join(MODELS_DIR, 'RF_Sleep.pkl')
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    logger.info(f"Random Forest model loaded from: {RF_MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading Random Forest model from {RF_MODEL_PATH}: {e}")
    rf_model = None

# --- Load the Quantized TFLite CNN Model ---
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'CNN_Sleep_quantized.tflite')
cnn_interpreter = None
try:
    # Load the TFLite model and allocate tensors.
    cnn_interpreter = tf.lite.Interpreter(model_path=CNN_MODEL_PATH)
    cnn_interpreter.allocate_tensors()
    logger.info(f"Quantized CNN TFLite model loaded from: {CNN_MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading quantized CNN TFLite model from {CNN_MODEL_PATH}: {e}")

# --- Load the Quantized TFLite LSTM Model ---
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'LSTM_Sleep_quantized.tflite')
lstm_interpreter = None
try:
    # Load the TFLite model and allocate tensors.
    lstm_interpreter = tf.lite.Interpreter(model_path=LSTM_MODEL_PATH)
    lstm_interpreter.allocate_tensors()
    logger.info(f"Quantized LSTM TFLite model loaded from: {LSTM_MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading quantized LSTM TFLite model from {LSTM_MODEL_PATH}: {e}")

logger.info("All models and preprocessors loading complete.")