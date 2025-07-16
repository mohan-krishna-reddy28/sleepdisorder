# FRONTEND/webapp/views.py
import os
import numpy as np
from django.shortcuts import render, redirect
from django import forms
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.conf import settings
import pandas as pd # Added pandas for robust data handling
import traceback # For detailed error logging

# Import the model_loader module
from . import model_loader

# Form class defined directly in views.py
class SleepForm(forms.Form):
    GENDER_CHOICES = [('0', 'Male'), ('1', 'Female')]
    OCCUPATION_CHOICES = [('0', 'Student'), ('1', 'Employee'), ('2', 'Self-employed')]
    BMI_CHOICES = [('0', 'Underweight'), ('1', 'Normal weight'), ('2', 'Overweight'), ('3', 'Obese')]
    BP_CHOICES = [('0', 'Low'), ('1', 'Normal'), ('2', 'High')]
    # REMOVED 'ANN' from ALGORITHM_CHOICES
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

# Mapping encoded values back to disorder types (ensure this matches your training's output classes)
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
            return render(request, 'index.html', {'error': 'Account file (account.txt) not found in FRONTEND folder!'})
        except Exception as e:
            return render(request, 'index.html', {'error': f'Error reading account file: {e}'})

        for account in accounts:
            if len(account) >= 2 and account[0] == name and account[1] == password:
                return redirect('base') # Redirect to the 'base' URL name

        return render(request, 'index.html', {'error': 'Wrong name or password'})

    return render(request, 'index.html') # The login page (or whatever your home template is)

def input(request): # Keep this function name as you provided
    form = SleepForm()
    return render(request, 'input.html', {'form': form})

def output(request): # Keep this function name as you provided, handles the prediction logic
    if request.method == 'POST':
        form = SleepForm(request.POST)
        if form.is_valid():
            prediction_text = "Error: Prediction could not be made." # Default error message
            model_used_name = "N/A" # Default model name

            try:
                loaded_models = model_loader.get_loaded_models()

                cnn_model = loaded_models.get('cnn_model')
                lstm_model = loaded_models.get('lstm_model')
                scaler = loaded_models.get('scaler')
                label_encoders = loaded_models.get('label_encoders')
                rf_model = loaded_models.get('rf_model')

                if scaler is None or label_encoders is None:
                    return HttpResponse("Prediction pre-processing resources (scaler/encoders) are not available. Check server logs for loading errors.", status=503)

                cleaned_data = form.cleaned_data
                processed_data_dict = {}

                form_categorical_fields_map = {
                    'gender': 'Gender',
                    'occupation': 'Occupation',
                    'bmi_category': 'BMI Category',
                    'blood_pressure': 'Blood Pressure'
                }

                for form_field_name, original_col_name in form_categorical_fields_map.items():
                    if form_field_name in cleaned_data:
                        value = cleaned_data[form_field_name]
                        if label_encoders and original_col_name in label_encoders:
                            le = label_encoders[original_col_name]
                            try:
                                processed_data_dict[form_field_name] = le.transform([str(value)])[0]
                            except ValueError:
                                # This means an unseen categorical value. You might need to handle this
                                # differently based on your model's robustness (e.g., using -1 or a default)
                                print(f"Warning: Unseen value '{value}' for {original_col_name}. Skipping encoding.")
                                processed_data_dict[form_field_name] = int(value) # Fallback to int if encoding fails
                        else:
                            print(f"Warning: LabelEncoder not found for {original_col_name}. Using raw value.")
                            processed_data_dict[form_field_name] = int(value)

                numerical_fields = [
                    'age', 'sleep_duration', 'quality_of_sleep',
                    'physical_activity_level', 'stress_level',
                    'heart_rate', 'daily_steps'
                ]
                for field in numerical_fields:
                    if field in cleaned_data:
                        processed_data_dict[field] = float(cleaned_data[field])

                # CRITICAL: This order MUST exactly match your training data's feature order
                # after all preprocessing (encoding and scaling). It's 11 features.
                feature_order = [
                    'gender', 'age', 'occupation', 'sleep_duration', 'quality_of_sleep',
                    'physical_activity_level', 'stress_level', 'bmi_category',
                    'blood_pressure', 'heart_rate', 'daily_steps'
                ]

                input_features = [processed_data_dict[f] for f in feature_order]
                data_for_scaler = np.array(input_features).reshape(1, -1)
                data_scaled = scaler.transform(data_for_scaler)

                selected_algorithm = cleaned_data['algorithm']
                prediction_value = None

                if selected_algorithm == 'RF':
                    if rf_model:
                        prediction_value = rf_model.predict(data_scaled)[0]
                        model_used_name = "Random Forest"
                    else:
                        prediction_text = "Random Forest Model is not available."
                elif selected_algorithm == 'CNN':
                    if cnn_model:
                        # Reshape for CNN (expects 3D input) - confirm (1, 11, 1) matches your CNN's input_shape
                        cnn_prediction = cnn_model.predict(data_scaled.reshape(1, 11, 1))
                        prediction_value = np.argmax(cnn_prediction)
                        model_used_name = "CNN"
                    else:
                        prediction_text = "CNN Model is not available."
                elif selected_algorithm == 'LSTM':
                    if lstm_model:
                        # Reshape for LSTM (expects 3D input) - confirm (1, 11, 1) matches your LSTM's input_shape
                        lstm_prediction = lstm_model.predict(data_scaled.reshape(1, 11, 1))
                        prediction_value = np.argmax(lstm_prediction)
                        model_used_name = "LSTM"
                    else:
                        prediction_text = "LSTM Model is not available."
                else:
                    prediction_text = "Selected algorithm model not loaded or invalid."

                if prediction_value is not None:
                    # Map the numerical prediction back to the human-readable disorder name
                    prediction_text = disorder_mapping.get(prediction_value, "Unknown Disorder Type")

            except Exception as e:
                traceback.print_exc() # This will print the full traceback to Render logs
                prediction_text = f"Prediction failed due to an internal server error: {e}"
                # You might want to return an actual HTTP error code for AJAX requests
                # return JsonResponse({'error': prediction_text}, status=500)

            return render(request, 'output.html', {
                'prediction': prediction_text,
                'model_used': model_used_name
            })
    return render(request, 'input.html', {'form': SleepForm()})

def about(request):
    return render(request, 'about.html')

def team(request):
    return render(request, 'team.html')

def base(request):
    return render(request, 'base.html')