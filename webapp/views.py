# FRONTEND/webapp/views.py
import os
import numpy as np
from django.shortcuts import render, redirect
from django import forms
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.conf import settings
import pandas as pd # Added pandas for robust data handling
import traceback # For detailed error logging
import tensorflow as tf # Added: Needed for TFLite operations

# Import the pre-loaded models and preprocessors from model_loader
# Make sure these are installed: pip install tensorflow scikit-learn pandas
from .model_loader import scaler, label_encoders, rf_model, cnn_interpreter, lstm_interpreter

# Form class defined directly in views.py
class SleepForm(forms.Form):
    GENDER_CHOICES = [('0', 'Male'), ('1', 'Female')]
    OCCUPATION_CHOICES = [('0', 'Student'), ('1', 'Employee'), ('2', 'Self-employed')]
    BMI_CHOICES = [('0', 'Underweight'), ('1', 'Normal weight'), ('2', 'Overweight'), ('3', 'Obese')]
    BP_CHOICES = [('0', 'Low'), ('1', 'Normal'), ('2', 'High')]
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
                # Check if essential pre-processing resources are available
                if scaler is None or label_encoders is None:
                    return HttpResponse("Prediction pre-processing resources (scaler/encoders) are not available. Check server logs for loading errors.", status=503)

                cleaned_data = form.cleaned_data
                processed_data_dict = {}

                # Map form field names to original column names for label encoders
                form_categorical_fields_map = {
                    'gender': 'Gender',
                    'occupation': 'Occupation',
                    'bmi_category': 'BMI Category',
                    'blood_pressure': 'Blood Pressure'
                }

                # Process categorical fields using loaded LabelEncoders
                for form_field_name, original_col_name in form_categorical_fields_map.items():
                    if form_field_name in cleaned_data:
                        value = cleaned_data[form_field_name]
                        if label_encoders and original_col_name in label_encoders:
                            le = label_encoders[original_col_name]
                            try:
                                # Ensure value is string for .transform()
                                processed_data_dict[form_field_name] = le.transform([str(value)])[0]
                            except ValueError:
                                print(f"Warning: Unseen value '{value}' for {original_col_name}. Skipping encoding.")
                                processed_data_dict[form_field_name] = int(value) # Fallback if unseen
                        else:
                            print(f"Warning: LabelEncoder not found for {original_col_name}. Using raw value.")
                            processed_data_dict[form_field_name] = int(value)

                # Process numerical fields
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

                # Prepare data for scaling
                input_features = [processed_data_dict[f] for f in feature_order]
                data_for_scaler = np.array(input_features).reshape(1, -1)
                # Scale data and ensure float32 type for TFLite models
                data_scaled = scaler.transform(data_for_scaler).astype(np.float32)

                selected_algorithm = cleaned_data['algorithm']
                prediction_value = None

                # --- Random Forest Prediction ---
                if selected_algorithm == 'RF':
                    if rf_model:
                        prediction_value = rf_model.predict(data_scaled)[0]
                        model_used_name = "Random Forest"
                    else:
                        prediction_text = "Random Forest Model is not available."
                # --- CNN (TFLite) Prediction ---
                elif selected_algorithm == 'CNN':
                    if cnn_interpreter:
                        input_details_cnn = cnn_interpreter.get_input_details()
                        output_details_cnn = cnn_interpreter.get_output_details()

                        # Debugging prints for CNN input shape
                        print(f"CNN Interpreter Expected Input Shape: {input_details_cnn[0]['shape']}")
                        print(f"CNN Input Data Shape before conversion: {data_scaled.shape}")

                        # Based on "Dimension mismatch. Got 3 but expected 2",
                        # the CNN TFLite model expects a 2D input.
                        # data_scaled is already (1, 11), which is 2D.
                        cnn_input_data = data_scaled # Use data_scaled directly (shape: (1, 11))

                        # Set input tensor and invoke the interpreter
                        cnn_interpreter.set_tensor(input_details_cnn[0]['index'], cnn_input_data)
                        cnn_interpreter.invoke()

                        # Get output tensor and process prediction
                        cnn_output_tensor = cnn_interpreter.get_tensor(output_details_cnn[0]['index'])
                        prediction_value = np.argmax(cnn_output_tensor, axis=1)[0]
                        model_used_name = "CNN"
                        print(f"CNN TFLite Prediction Output (Raw): {cnn_output_tensor}")
                        print(f"CNN TFLite Predicted Class Index: {prediction_value}")
                    else:
                        prediction_text = "CNN Model (TFLite interpreter) is not available."
                # --- LSTM (TFLite) Prediction ---
                elif selected_algorithm == 'LSTM':
                    if lstm_interpreter:
                        input_details_lstm = lstm_interpreter.get_input_details()
                        output_details_lstm = lstm_interpreter.get_output_details()

                        required_lstm_shape = input_details_lstm[0]['shape']
                        print(f"LSTM Interpreter Expected Input Shape: {required_lstm_shape}")
                        print(f"LSTM Input Data Shape before conversion: {data_scaled.shape}")

                        lstm_input_data = data_scaled # Start with scaled features (1, 11)

                        # Adjust LSTM input shape if necessary, typically expects 3D input (batch, timesteps, features)
                        if len(required_lstm_shape) == 3:
                            # Reshape to (1, 11, 1) if your LSTM accepts 11 timesteps with 1 feature each,
                            # or (1, 1, 11) if it accepts 1 timestep with 11 features.
                            # Assuming (1, 11, 1) from common practice for 1D sequence models
                            lstm_input_data = data_scaled.reshape(1, data_scaled.shape[1], 1)
                            print(f"LSTM Input Data Reshaped to: {lstm_input_data.shape}")
                        else:
                            # If LSTM also expects 2D (like for a simple Dense layer or a flattened input),
                            # data_scaled is already (1, 11)
                            print("LSTM Interpreter expects 2D input. Using data_scaled directly.")

                        # Set input tensor and invoke the interpreter
                        lstm_interpreter.set_tensor(input_details_lstm[0]['index'], lstm_input_data)
                        lstm_interpreter.invoke()

                        # Get output tensor and process prediction
                        lstm_output_tensor = lstm_interpreter.get_tensor(output_details_lstm[0]['index'])
                        prediction_value = np.argmax(lstm_output_tensor, axis=1)[0]
                        model_used_name = "LSTM"
                        print(f"LSTM TFLite Prediction Output (Raw): {lstm_output_tensor}")
                        print(f"LSTM TFLite Predicted Class Index: {prediction_value}")
                    else:
                        prediction_text = "LSTM Model (TFLite interpreter) is not available."
                # --- Fallback for invalid algorithm selection ---
                else:
                    prediction_text = "Selected algorithm model not loaded or invalid."

                if prediction_value is not None:
                    # Map the numerical prediction back to the human-readable disorder name
                    prediction_text = disorder_mapping.get(prediction_value, "Unknown Disorder Type")

            except Exception as e:
                traceback.print_exc() # This will print the full traceback to the server logs
                prediction_text = f"Prediction failed due to an internal server error: {e}"
                # For AJAX requests, you might return JsonResponse({'error': prediction_text}, status=500)

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