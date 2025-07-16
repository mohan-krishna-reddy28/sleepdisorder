import os
import numpy as np
from django.shortcuts import render, redirect
from django import forms
from django.http import HttpResponse
from django.conf import settings

# Import the new model_loader module
from . import model_loader # <--- NEW: Import your model_loader.py

# Form class defined directly in views.py
class SleepForm(forms.Form):
    GENDER_CHOICES = [('0', 'Male'), ('1', 'Female')]
    OCCUPATION_CHOICES = [('0', 'Student'), ('1', 'Employee'), ('2', 'Self-employed')]
    BMI_CHOICES = [('0', 'Underweight'), ('1', 'Normal weight'), ('2', 'Overweight'), ('3', 'Obese')]
    BP_CHOICES = [('0', 'Low'), ('1', 'Normal'), ('2', 'High')]
    # --- UPDATED: Include all algorithms, including ANN ---
    ALGORITHM_CHOICES = [('RF', 'Random Forest'), ('CNN', 'CNN'), ('LSTM', 'LSTM'), ('ANN', 'ANN')]

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

# Mapping encoded values back to disorder types
disorder_mapping = {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}

def home(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        try:
            # Path to account.txt is relative to settings.BASE_DIR (FRONTEND folder)
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
            prediction = "Error: Model not found" # Default error message

            try:
                # --- NEW: Get all loaded models and preprocessors from model_loader ---
                loaded_models = model_loader.get_loaded_models()

                # Assign them to local variables for easier use
                cnn_model = loaded_models.get('cnn_model')
                lstm_model = loaded_models.get('lstm_model')
                ann_model = loaded_models.get('ann_model') # Get the ANN model
                scaler = loaded_models.get('scaler')
                label_encoders = loaded_models.get('label_encoders')
                rf_model = loaded_models.get('rf_model')

                # --- Validate that essential preprocessors are loaded ---
                if scaler is None or label_encoders is None:
                    return HttpResponse("Prediction pre-processing resources (scaler/encoders) are not available. Check server logs for loading errors.", status=503)

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
                        if label_encoders and original_col_name in label_encoders:
                            processed_data[key] = label_encoders[original_col_name].transform([str(value)])[0]
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

                # Apply the scaler to the numerical features
                data_scaled = scaler.transform(data)

                selected_algorithm = form.cleaned_data['algorithm']

                # --- Prediction Logic using loaded_models ---
                if selected_algorithm == 'RF':
                    if rf_model:
                        prediction = disorder_mapping[rf_model.predict(data_scaled)[0]]
                    else:
                        prediction = "Random Forest Model is not available."
                elif selected_algorithm == 'CNN':
                    if cnn_model:
                        # Reshape for CNN (expects 3D input)
                        cnn_prediction = cnn_model.predict(data_scaled.reshape(1, 11, 1))
                        prediction = disorder_mapping[np.argmax(cnn_prediction)]
                    else:
                        prediction = "CNN Model is not available."
                elif selected_algorithm == 'LSTM':
                    if lstm_model:
                        # Reshape for LSTM (expects 3D input)
                        lstm_prediction = lstm_model.predict(data_scaled.reshape(1, 11, 1))
                        prediction = disorder_mapping[np.argmax(lstm_prediction)]
                    else:
                        prediction = "LSTM Model is not available."
                elif selected_algorithm == 'ANN': # <--- NEW: ANN Prediction Logic
                    if ann_model:
                        # Reshape for ANN (expects 2D input if Dense layer input_dim=11)
                        ann_prediction = ann_model.predict(data_scaled)
                        prediction = disorder_mapping[np.argmax(ann_prediction)]
                    else:
                        prediction = "ANN Model is not available."
                else:
                    prediction = "Selected algorithm model not loaded or invalid."

            except Exception as e:
                # Catch any errors during model retrieval or prediction
                print(f"Error during prediction process: {e}")
                prediction = f"Prediction error: {e}"
                # Optionally, return an HTTP error response instead of rendering
                # return HttpResponse(f"Server error during prediction: {e}", status=500)

            return render(request, 'output.html', {'prediction': prediction})
    return render(request, 'input.html', {'form': SleepForm()})

def about(request):
    return render(request, 'about.html')

def team(request):
    return render(request, 'team.html')

def base(request):
    return render(request, 'base.html')