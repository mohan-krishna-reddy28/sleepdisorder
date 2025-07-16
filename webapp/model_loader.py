# FRONTEND/webapp/model_loader.py
import os
import joblib
import tensorflow as tf
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

_models = {} # Dictionary to store loaded models

def _get_model_path(filename):
    """
    Helper to construct absolute path to model files.
    Looks for models in the parent directory of this model_loader.py.
    (i.e., from webapp/ go up to FRONTEND/)
    """
    # Current directory of model_loader.py is FRONTEND/webapp/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to FRONTEND/ (where your models are)
    models_dir = os.path.dirname(current_dir)
    return os.path.join(models_dir, filename)

def load_model_and_preprocessors():
    if not _models: # If _models is empty, it means they haven't been loaded yet
        logger.info("Starting to load machine learning models and preprocessors...")
        try:
            # Load Keras models (CNN and LSTM)
            # Ensure these filenames match exactly what you saved: CNN_Sleep.h5, LSTM_Sleep.h5
            _models['cnn_model'] = tf.keras.models.load_model(_get_model_path('CNN_Sleep.h5'))
            logger.info("CNN model loaded.")

            _models['lstm_model'] = tf.keras.models.load_model(_get_model_path('LSTM_Sleep.h5'))
            logger.info("LSTM model loaded.")

            # Load Scikit-learn models/preprocessors (Scaler, Label Encoders, Random Forest)
            # Ensure these filenames match exactly what you saved: scaler.pkl, label_encoders.pkl, RF_Sleep.pkl
            _models['scaler'] = joblib.load(_get_model_path('scaler.pkl'))
            logger.info("Scaler loaded.")

            _models['label_encoders'] = joblib.load(_get_model_path('label_encoders.pkl'))
            logger.info("Label encoders loaded.")

            _models['rf_model'] = joblib.load(_get_model_path('RF_Sleep.pkl'))
            logger.info("Random Forest model loaded.")

            logger.info("All essential models and preprocessors loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading models or preprocessors: {e}", exc_info=True)
            raise # Re-raise the error to ensure it's visible in Render logs

    return _models

def get_loaded_models():
    if not _models:
        logger.warning("Models not loaded yet. Calling load_model_and_preprocessors().")
        load_model_and_preprocessors()
    return _models

try:
    # Attempt to load models at app startup (on the first worker)
    # This is where the heavy lifting happens, potentially causing timeout on free tier
    load_model_and_preprocessors()
except Exception:
    # Catch and pass if loading fails at startup (error will be logged by the loader function itself)
    pass