import os
import joblib
import numpy as np
from django.shortcuts import render, redirect
from django import forms
from django.http import HttpResponse
from django.conf import settings # <-- THIS IMPORT IS CRUCIAL

# For Keras model loading
from tensorflow.keras.models import load_model as keras_load_model

# Form class defined directly in views.py
class SleepForm(forms.Form):
    GENDER_CHOICES = [('0', 'Male'), ('1', 'Female')]
    OCCUPATION_CHOICES = [('0', 'Student'), ('1', 'Employee'), ('2', 'Self-employed')]
    BMI_CHOICES = [('0', 'Underweight'), ('1', 'Normal weight'), ('2', 'Overweight'), ('3', 'Obese')]
    BP_CHOICES = [('0', 'Low'), ('1', 'Normal'), ('2', 'High')]
    ALGORITHM_CHOICES = [('CNN', 'CNN')]

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
global_label_encoders = None
global_rf_model = None # Placeholder for RF model
global_lstm_model = None # Placeholder for LSTM model

def load_ml_assets():
    """
    Loads the machine learning model and preprocessors if they haven't been loaded yet.
    This function will be called only when a prediction is needed.
    """
    global global_cnn_model, global_scaler, global_label_encoders, global_rf_model, global_lstm_model

    # Only load if the CNN model (or any other main asset) is not already in memory
    if global_cnn_model is None:
        print("Attempting to load ML assets for the first time...")
        try:
            # Define the directory where your model files are stored.
            # It's best practice to put them in a dedicated folder, e.g., 'model_files'
            # at the root of your Django project (where manage.py is).
            MODEL_FILES_DIR = os.path.join(settings.BASE_DIR, 'model_files')

            CNN_MODEL_PATH = os.path.join(MODEL_FILES_DIR, 'cnn_model.h5')
            SCALER_PATH = os.path.join(MODEL_FILES_DIR, 'scaler_n.pkl')
            LABEL_ENCODERS_PATH = os.path.join(MODEL_FILES_DIR, 'label_encoders.pkl')

            # Basic checks to ensure files exist
            if not os.path.exists(CNN_MODEL_PATH):
                raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
            if not os.path.exists(LABEL_ENCODERS_PATH):
                raise FileNotFoundError(f"Label encoders not found: {LABEL_ENCODERS_PATH}")

            global_cnn_model = keras_load_model(CNN_MODEL_PATH)
            global_scaler = joblib.load(SCALER_PATH)
            global_label_encoders = joblib.load(LABEL_ENCODERS_PATH)

            # Assign placeholders for other models (load them if you save them later)
            global_rf_model = None
            global_lstm_model = None

            print("ML assets loaded successfully!")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load ML assets: {e}")
            # Reset globals to None if loading fails
            global_cnn_model = None
            global_scaler = None
            global_label_encoders = None
            global_rf_model = None
            global_lstm_model = None
            # Re-raise the exception to indicate a problem to the calling view
            raise

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
                return redirect('base')

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

            # Ensure models are loaded before proceeding
            if global_cnn_model is None or global_scaler is None or global_label_encoders is None:
                return HttpResponse("Prediction resources are not available.", status=503)

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
                    if original_col_name in global_label_encoders:
                        processed_data[key] = global_label_encoders[original_col_name].transform([str(value)])[0]
                    else:
                        print(f"Warning: LabelEncoder not found for {original_col_name}. Using raw value.")
                        processed_data[key] = int(value)
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

            selected_algorithm = form.cleaned_data['algorithm']
            prediction = "Error: Model not found"

            # Use global_cnn_model, global_rf_model, global_lstm_model
            if selected_algorithm == 'RF' and global_rf_model:
                prediction = disorder_mapping[global_rf_model.predict(data)[0]]
            elif selected_algorithm == 'CNN' and global_cnn_model:
                cnn_prediction = global_cnn_model.predict(data.reshape(1, 11, 1))
                prediction = disorder_mapping[np.argmax(cnn_prediction)]
            elif selected_algorithm == 'LSTM' and global_lstm_model:
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