import os # <-- ADD THIS IMPORT
import joblib # <-- ADD THIS IMPORT for .pkl files
import numpy as np
from django.shortcuts import render, redirect # <-- Corrected imports
from django import forms
from django.http import HttpResponse

# For Keras model loading
from tensorflow.keras.models import load_model as keras_load_model # Renamed to avoid conflict

# Form class defined directly in views.py
class SleepForm(forms.Form):
    GENDER_CHOICES = [('0', 'Male'), ('1', 'Female')]
    OCCUPATION_CHOICES = [('0', 'Student'), ('1', 'Employee'), ('2', 'Self-employed')]
    BMI_CHOICES = [('0', 'Underweight'), ('1', 'Normal weight'), ('2', 'Overweight'), ('3', 'Obese')]
    BP_CHOICES = [('0', 'Low'), ('1', 'Normal'), ('2', 'High')]
    ALGORITHM_CHOICES = [('CNN', 'CNN')] # <-- Adjusted choices, as we only have CNN saved and ready
    # ALGORITHM_CHOICES = [('RF', 'Random Forest'), ('CNN', 'CNN'), ('LSTM', 'LSTM')] # Original, if you add RF/LSTM later

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

# --- START: CORRECTED MODEL AND PREPROCESSOR LOADING ---

# Get the directory of the current file (views.py) dynamically
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to your saved files relative to the 'webapp' directory.
# These files should be directly inside the 'webapp' folder on Render.
CNN_MODEL_PATH = os.path.join(CURRENT_DIR, 'cnn_model.h5') # Your Keras model
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler_n.pkl') # Your StandardScaler
LABEL_ENCODERS_PATH = os.path.join(CURRENT_DIR, 'label_encoders.pkl') # Your LabelEncoders dictionary

# Load the model and preprocessors
try:
    # Load the Keras CNN model (.h5 file) using keras_load_model
    cnn_model = keras_load_model(CNN_MODEL_PATH)

    # Load the StandardScaler (.pkl file) using joblib
    scaler = joblib.load(SCALER_PATH)

    # Load the LabelEncoders dictionary (.pkl file) using joblib
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)

    # Commenting out RF and LSTM loading as these were not saved in your notebook
    # If you later train and save these, you will uncomment and update their paths/loading methods.
    rf_model = None # Placeholder, or load if you have RF_Sleep.pkl saved via joblib/pickle
    lstm_model = None # Placeholder, or load if you have LSTM_Sleep.pkl saved via joblib/pickle

    print("Model and preprocessors loaded successfully!")
except Exception as e:
    print(f"Error loading model or preprocessors: {e}")
    # Set to None if loading fails to prevent errors later
    cnn_model = None
    scaler = None
    label_encoders = None
    rf_model = None
    lstm_model = None

# Mapping encoded values back to disorder types
disorder_mapping = {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}

# --- END: CORRECTED MODEL AND PREPROCESSOR LOADING ---


def home(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        try:
            # Assuming 'account.txt' is also in the webapp directory for Render
            account_file_path = os.path.join(CURRENT_DIR, 'account.txt')
            with open(account_file_path, 'r') as f:
                accounts = [line.strip().split() for line in f.readlines()]
        except FileNotFoundError:
            return render(request, 'index.html', {'error': 'Account file not found!'})

        for account in accounts:
            if len(account) >= 2 and account[0] == name and account[1] == password:
                return redirect('base') # Redirect to 'base' instead of 'input' directly

        return render(request, 'index.html', {'error': 'Wrong name or password'})

    return render(request, 'index.html')


def input(request):
    form = SleepForm()
    return render(request, 'input.html', {'form': form})


def output(request):
    if request.method == 'POST':
        form = SleepForm(request.POST)
        if form.is_valid():
            # Collect data, applying label encoding to categorical fields
            cleaned_data = form.cleaned_data
            processed_data = {}

            # List of categorical columns that need encoding from the form
            # Ensure these match the order and names expected by your scaler/model after encoding
            form_categorical_cols = [
                'gender', 'occupation', 'bmi_category', 'blood_pressure'
            ]
            # Map form field names to the original column names used during training if they differ
            # For example: 'gender' in form maps to 'Gender' in label_encoders keys
            column_name_map = {
                'gender': 'Gender',
                'occupation': 'Occupation',
                'bmi_category': 'BMI Category',
                'blood_pressure': 'Blood Pressure'
            }


            for key, value in cleaned_data.items():
                if key in form_categorical_cols:
                    original_col_name = column_name_map.get(key, key.capitalize()) # Default to capitalize if no explicit map
                    if original_col_name in label_encoders:
                        # Convert input string choice to integer index using the loaded LabelEncoder
                        processed_data[key] = label_encoders[original_col_name].transform([str(value)])[0]
                    else:
                        print(f"Warning: LabelEncoder not found for {original_col_name}")
                        processed_data[key] = int(value) # Fallback to int if not found
                else:
                    processed_data[key] = float(value) if isinstance(value, (int, float)) else int(value) # Keep numerical as is, convert to int if string

            # Create numpy array in the exact order expected by the model
            # This order must match the order of features used during model training (X array)
            # You might need to adjust this list based on your notebook's X_train feature order
            feature_order = [
                'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
                'physical_activity_level', 'stress_level', 'bmi_category',
                'blood_pressure', 'heart_rate', 'daily_steps'
            ]

            # Convert processed_data dictionary to a list in the correct order
            input_features = [processed_data[f] for f in feature_order]

            data = np.array(input_features).reshape(1, -1) # Reshape for single prediction

            # Apply the scaler to the numerical features
            data = scaler.transform(data)

            selected_algorithm = form.cleaned_data['algorithm']
            prediction = "Error: Model not found" # Default prediction

            if selected_algorithm == 'RF' and rf_model:
                prediction = disorder_mapping[rf_model.predict(data)[0]]
            elif selected_algorithm == 'CNN' and cnn_model:
                cnn_prediction = cnn_model.predict(data.reshape(1, 11, 1)) # Reshape for CNN input
                prediction = disorder_mapping[np.argmax(cnn_prediction)]
            elif selected_algorithm == 'LSTM' and lstm_model:
                prediction = disorder_mapping[np.argmax(lstm_model.predict(data.reshape(1, 11, 1)))]
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