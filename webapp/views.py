import os
import joblib
import numpy as np
from django.shortcuts import render, redirect
from django import forms
from django.http import HttpResponse
from django.conf import settings

# For Keras model loading
# IMPORTANT: Keras models (CNN, LSTM) are typically saved as .h5 or .keras, not .pkl.
# If your CNN_Sleep.pkl and LSTM_Sleep.pkl are truly Keras models, you should re-save them as .h5
# and use keras_load_model. For now, I'm assuming they are loaded via joblib.load()
# as their extension is .pkl. If this causes errors, you MUST check your model saving process.
from tensorflow.keras.models import load_model as keras_load_model

# Form class defined directly in views.py
class SleepForm(forms.Form):
    GENDER_CHOICES = [('0', 'Male'), ('1', 'Female')]
    OCCUPATION_CHOICES = [('0', 'Student'), ('1', 'Employee'), ('2', 'Self-employed')]
    BMI_CHOICES = [('0', 'Underweight'), ('1', 'Normal weight'), ('2', 'Overweight'), ('3', 'Obese')]
    BP_CHOICES = [('0', 'Low'), ('1', 'Normal'), ('2', 'High')]
    # --- CORRECTED: Include all algorithms from your UI ---
    ALGORITHM_CHOICES = [('RF', 'Random Forest'), ('CNN', 'CNN'), ('LSTM', 'LSTM')]

    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    age = forms.IntegerField()
    occupation = forms.ChoiceField(choices=OCCUPATION_CHOICES)
    sleep_duration = forms.FloatField()
    quality_of_sleep = forms.IntegerField()
    physical_activity_level = forms.IntegerField()
    stress_level = forms.IntegerField()
    bmi_category = forms.ChoiceField(choices=BMI_CHOICES)
    blood_pressure = forms.ChoiceField(choices=BP_CHOICES)
    heart_rate = forms.IntegerField()
    daily_steps = forms.IntegerField()
    algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES)

# --- START: LAZY LOADING MODEL AND PREPROCESSORS ---

# Global variables to hold the loaded model and preprocessors.
# They are initialized to None.
global_cnn_model = None
global_scaler = None
global_label_encoders = None # Assuming you have a label_encoders.pkl file at the root
global_rf_model = None
global_lstm_model = None

def load_ml_assets():
    """
    Loads the machine learning model and preprocessors if they haven't been loaded yet.
    This function will be called only when a prediction is needed.
    """
    global global_cnn_model, global_scaler, global_label_encoders, global_rf_model, global_lstm_model

    # Only load if models haven't been loaded yet (e.g., first request)
    if global_cnn_model is None or global_rf_model is None or global_lstm_model is None:
        print("Attempting to load ML assets for the first time...")
        try:
            # --- CORRECTED PATHS AND FILENAMES based on your project structure ---
            # All files are directly in BASE_DIR (project root), not in a 'model_files' subfolder.
            CNN_MODEL_PATH = os.path.join(settings.BASE_DIR, 'CNN_Sleep.pkl') # Corrected filename and path
            SCALER_PATH = os.path.join(settings.BASE_DIR, 'scaler.pkl')       # Corrected filename and path
            LABEL_ENCODERS_PATH = os.path.join(settings.BASE_DIR, 'label_encoders.pkl') # Assuming this exists at root
            RF_MODEL_PATH = os.path.join(settings.BASE_DIR, 'RF_Sleep.pkl')   # Corrected filename and path
            LSTM_MODEL_PATH = os.path.join(settings.BASE_DIR, 'LSTM_Sleep.pkl') # Corrected filename and path

            # Basic checks to ensure files exist and provide warnings if not
            if not os.path.exists(CNN_MODEL_PATH):
                raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
            # Optional: Check for label_encoders.pkl, and initialize if not found
            if not os.path.exists(LABEL_ENCODERS_PATH):
                print(f"Warning: Label encoders not found at {LABEL_ENCODERS_PATH}. Initializing empty dict.")
                global_label_encoders = {} # Provide a fallback to prevent KeyError later
            if not os.path.exists(RF_MODEL_PATH):
                print(f"Warning: RF model not found at {RF_MODEL_PATH}. RF predictions will not be available.")
            if not os.path.exists(LSTM_MODEL_PATH):
                print(f"Warning: LSTM model not found at {LSTM_MODEL_PATH}. LSTM predictions will not be available.")

            # --- Model Loading ---
            # IMPORTANT: Keras models (CNN, LSTM) typically use .h5 or .keras extension.
            # If your CNN_Sleep.pkl and LSTM_Sleep.pkl are Keras models saved as .pkl,
            # this is unconventional and might lead to issues. They might need to be
            # re-saved as .h5 and loaded with `keras_load_model`.
            # For now, we attempt to load all .pkl files using joblib.load().

            try:
                # If CNN_Sleep.pkl is a Keras model, you might need to change this
                # to keras_load_model(CNN_MODEL_PATH) after saving it as .h5
                global_cnn_model = joblib.load(CNN_MODEL_PATH)
                print(f"Successfully loaded CNN model from {CNN_MODEL_PATH}")
            except Exception as e:
                print(f"ERROR: Failed to load CNN model from {CNN_MODEL_PATH}: {e}")
                global_cnn_model = None

            try:
                global_scaler = joblib.load(SCALER_PATH)
                print(f"Successfully loaded scaler from {SCALER_PATH}")
            except Exception as e:
                print(f"ERROR: Failed to load scaler from {SCALER_PATH}: {e}")
                global_scaler = None

            # Load label encoders if the file exists
            if os.path.exists(LABEL_ENCODERS_PATH):
                try:
                    global_label_encoders = joblib.load(LABEL_ENCODERS_PATH)
                    print(f"Successfully loaded label encoders from {LABEL_ENCODERS_PATH}")
                except Exception as e:
                    print(f"ERROR: Failed to load label encoders from {LABEL_ENCODERS_PATH}: {e}")
                    global_label_encoders = {} # Fallback

            try:
                global_rf_model = joblib.load(RF_MODEL_PATH)
                print(f"Successfully loaded RF model from {RF_MODEL_PATH}")
            except Exception as e:
                print(f"ERROR: Failed to load RF model from {RF_MODEL_PATH}: {e}")
                global_rf_model = None

            try:
                # If LSTM_Sleep.pkl is a Keras model, you might need to change this
                # to keras_load_model(LSTM_MODEL_PATH) after saving it as .h5
                global_lstm_model = joblib.load(LSTM_MODEL_PATH)
                print(f"Successfully loaded LSTM model from {LSTM_MODEL_PATH}")
            except Exception as e:
                print(f"ERROR: Failed to load LSTM model from {LSTM_MODEL_PATH}: {e}")
                global_lstm_model = None

            print("ML assets loading attempt completed.")

        except Exception as e:
            # This outer catch is for critical errors like BASE_DIR not accessible,
            # or fundamental problems preventing any loading attempt.
            print(f"CRITICAL ERROR during initial ML assets loading: {e}")
            # Ensure all globals are reset if a critical error occurs
            global_cnn_model = None
            global_scaler = None
            global_label_encoders = None
            global_rf_model = None
            global_lstm_model = None
            raise # Re-raise to show the error on the webpage


# Mapping encoded values back to disorder types
disorder_mapping = {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}

# --- END: LAZY LOADING MODEL AND PREPROCESSORS ---

def home(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        try:
            # CORRECTED PATH: Now points directly to the project root for account.txt
            account_file_path = os.path.join(settings.BASE_DIR, 'account.txt')
            with open(account_file_path, 'r') as f:
                accounts = [line.strip().split() for line in f.readlines()]
        except FileNotFoundError:
            return render(request, 'index.html', {'error': 'Account file not found!'})

        for account in accounts:
            if len(account) >= 2 and account[0] == name and account[1] == password:
                return redirect('base') # Assuming 'base' is your input page URL name

        return render(request, 'index.html', {'error': 'Wrong name or password'})

    return render(request, 'index.html')


def input(request):
    form = SleepForm()
    return render(request, 'input.html', {'form': form})


def output(request):
    if request.method == 'POST':
        form = SleepForm(request.POST)
        if form.is_valid():
            # --- IMPORTANT: Call load_ml_assets() here! ---
            try:
                load_ml_assets() # This will load models only once across requests
            except Exception as e:
                # If loading fails, return an error message
                return HttpResponse(f"Server error: Could not load prediction resources. Details: {e}", status=500)

            # Ensure models are loaded before proceeding based on selected algorithm
            selected_algorithm = form.cleaned_data['algorithm']
            
            # Check if the specifically selected model is loaded
            if selected_algorithm == 'CNN' and global_cnn_model is None:
                 return HttpResponse("CNN Model is not available. Check server logs for loading errors.", status=503)
            elif selected_algorithm == 'RF' and global_rf_model is None:
                 return HttpResponse("Random Forest Model is not available. Check server logs for loading errors.", status=503)
            elif selected_algorithm == 'LSTM' and global_lstm_model is None:
                 return HttpResponse("LSTM Model is not available. Check server logs for loading errors.", status=503)
            
            if global_scaler is None or global_label_encoders is None:
                return HttpResponse("Prediction pre-processing resources (scaler/encoders) are not available.", status=503)


            # Collect data, applying label encoding to categorical fields
            cleaned_data = form.cleaned_data
            processed_data = {}

            form_categorical_cols = [
                'gender', 'occupation', 'bmi_category', 'blood_pressure'
            ]

            column_name_map = {
                'gender': 'Gender',
                'occupation': 'Occupation',
                'bmi_category': 'BMI Category',
                'blood_pressure': 'Blood Pressure'
            }

            for key, value in cleaned_data.items():
                if key in form_categorical_cols:
                    original_col_name = column_name_map.get(key, key.capitalize())
                    # Use global_label_encoders here
                    if global_label_encoders and original_col_name in global_label_encoders:
                        processed_data[key] = global_label_encoders[original_col_name].transform([str(value)])[0]
                    else:
                        print(f"Warning: LabelEncoder not found for {original_col_name}. Using raw value.")
                        processed_data[key] = int(value) # Fallback to raw int value
                else:
                    processed_data[key] = float(value) if isinstance(value, (int, float)) else int(value)

            feature_order = [
                'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
                'physical_activity_level', 'stress_level', 'bmi_category',
                'blood_pressure', 'heart_rate', 'daily_steps'
            ]

            input_features = [processed_data[f] for f in feature_order]

            data = np.array(input_features).reshape(1, -1)

            # Apply the scaler to the numerical features using global_scaler
            data = global_scaler.transform(data)

            prediction = "Error: Model not found" # Default error message

            # Use global_cnn_model, global_rf_model, global_lstm_model
            if selected_algorithm == 'RF' and global_rf_model:
                prediction = disorder_mapping[global_rf_model.predict(data)[0]]
            elif selected_algorithm == 'CNN' and global_cnn_model:
                # Reshape for CNN if needed (based on your original code, it expects 3D input)
                cnn_prediction = global_cnn_model.predict(data.reshape(1, 11, 1))
                prediction = disorder_mapping[np.argmax(cnn_prediction)]
            elif selected_algorithm == 'LSTM' and global_lstm_model:
                # Reshape for LSTM if needed (based on your original code, it expects 3D input)
                prediction = disorder_mapping[np.argmax(global_lstm_model.predict(data.reshape(1, 11, 1)))]
            else:
                prediction = "Selected algorithm model not loaded or invalid."

            return render(request, 'output.html', {'prediction': prediction})
    return render(request, 'input.html', {'form': SleepForm()})

def about(request):
    return render(request, 'about.html')

def team(request):
    return render(request, 'team.html')

def base(request):
    return render(request, 'base.html')