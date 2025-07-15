# # views.py
# import pickle
# import numpy as np
# from django.shortcuts import render
# from django import forms
# from django.http import HttpResponse

# # Form class defined directly in views.py
# class SleepForm(forms.Form):
#     GENDER_CHOICES = [('0', 'Male'), ('1', 'Female')]
#     OCCUPATION_CHOICES = [('0', 'Student'), ('1', 'Employee'), ('2', 'Self-employed')]
#     BMI_CHOICES = [('0', 'Underweight'), ('1', 'Normal weight'), ('2', 'Overweight'), ('3', 'Obese')]
#     BP_CHOICES = [('0', 'Low'), ('1', 'Normal'), ('2', 'High')]
#     ALGORITHM_CHOICES = [('RF', 'Random Forest'), ('CNN', 'CNN'), ('LSTM', 'LSTM')]

#     gender = forms.ChoiceField(choices=GENDER_CHOICES)
#     age = forms.IntegerField()
#     occupation = forms.ChoiceField(choices=OCCUPATION_CHOICES)
#     sleep_duration = forms.FloatField()
#     quality_of_sleep = forms.IntegerField()
#     physical_activity_level = forms.IntegerField()
#     stress_level = forms.IntegerField()
#     bmi_category = forms.ChoiceField(choices=BMI_CHOICES)
#     blood_pressure = forms.ChoiceField(choices=BP_CHOICES)
#     heart_rate = forms.IntegerField()
#     daily_steps = forms.IntegerField()
#     algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES)

# def load_model(model_path):
#     with open(model_path, 'rb') as f:
#         return pickle.load(f)

# def read_file(file_name):
#     with open(file_name, 'r') as opened_file:
#         lines_list = [line.split() for line in opened_file]
#     return lines_list

# # Load models and scaler with Mohan's path
# rf_model = load_model(r'C:\Users\Mohan\OneDrive\Desktop\SLEEP_DISORDER\FRONTEND\RF_Sleep.pkl')
# cnn_model = load_model(r'C:\Users\Mohan\OneDrive\Desktop\SLEEP_DISORDER\FRONTEND\CNN_Sleep.pkl')
# lstm_model = load_model(r'C:\Users\Mohan\OneDrive\Desktop\SLEEP_DISORDER\FRONTEND\LSTM_Sleep.pkl')
# scaler = load_model(r'C:\Users\Mohan\OneDrive\Desktop\SLEEP_DISORDER\FRONTEND\scaler.pkl')

# # Mapping encoded values back to disorder types
# disorder_mapping = {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}

# def home(request):
#     return render(request, 'index.html')


# def team(request):
#     return render(request, 'team.html')

# def about(request):
#     return render(request, 'about.html')


# def input(request):
#     file_name = 'account.txt'
#     name = request.POST.get('name')
#     password = request.POST.get('password')
#     account_list = read_file(file_name)

#     print(f"Name: {name}, Password: {password}")
#     print("Accounts loaded:", account_list)

#     for i in account_list:
#         if i[0].strip().lower() == name.strip().lower() and i[1].strip() == password.strip():
#             print("Login successful")
#             form = SleepForm()
#             return render(request, 'input.html', {'form': form})

#     return HttpResponse('Wrong Password or Name', content_type='text/plain')


# def output(request):
#     if request.method == 'POST':
#         form = SleepForm(request.POST)
#         if form.is_valid():
#             data = np.array([
#                 form.cleaned_data['gender'],
#                 form.cleaned_data['age'],
#                 form.cleaned_data['occupation'],
#                 form.cleaned_data['sleep_duration'],
#                 form.cleaned_data['quality_of_sleep'],
#                 form.cleaned_data['physical_activity_level'],
#                 form.cleaned_data['stress_level'],
#                 form.cleaned_data['bmi_category'],
#                 form.cleaned_data['blood_pressure'],
#                 form.cleaned_data['heart_rate'],
#                 form.cleaned_data['daily_steps']
#             ]).reshape(1, -1)

#             data = scaler.transform(data)

#             selected_algorithm = form.cleaned_data['algorithm']
#             if selected_algorithm == 'RF':
#                 prediction = disorder_mapping[rf_model.predict(data)[0]]
#             elif selected_algorithm == 'CNN':
#                 # prediction = disorder_mapping[np.argmax(cnn_model.predict(data))]
#                 cnn_prediction = cnn_model.predict(data.reshape(1, 11, 1))
#                 prediction = disorder_mapping[np.argmax(cnn_prediction)]
#             elif selected_algorithm == 'LSTM':
#                 prediction = disorder_mapping[np.argmax(lstm_model.predict(data.reshape(1, 11, 1)))]

#             return render(request, 'output.html', {
#                 'prediction': prediction,
#                 'confidence_score': 'N/A'  # Add logic if your models provide confidence scores
#             })
#     # return render(request, 'input.html', {'form': SleepForm()})
#     return render(request, 'input.html', {'form': form})



import pickle
import numpy as np
from django.shortcuts import render
from django import forms
from django.http import HttpResponse
from django.shortcuts import render, redirect

# Form for prediction
class SleepForm(forms.Form):
    gender = forms.ChoiceField(choices=[('0', 'Male'), ('1', 'Female')])
    age = forms.IntegerField()
    occupation = forms.ChoiceField(choices=[('0', 'Student'), ('1', 'Employee'), ('2', 'Self-employed')])
    sleep_duration = forms.FloatField()
    quality_of_sleep = forms.IntegerField()
    physical_activity_level = forms.IntegerField()
    stress_level = forms.IntegerField()
    bmi_category = forms.ChoiceField(choices=[('0', 'Underweight'), ('1', 'Normal'), ('2', 'Overweight'), ('3', 'Obese')])
    blood_pressure = forms.ChoiceField(choices=[('0', 'Low'), ('1', 'Normal'), ('2', 'High')])
    heart_rate = forms.IntegerField()
    daily_steps = forms.IntegerField()
    algorithm = forms.ChoiceField(choices=[('RF', 'Random Forest'), ('CNN', 'CNN'), ('LSTM', 'LSTM')])

# Load models
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def read_file(file_name):
    with open(file_name, 'r') as f:
        return [line.strip().split() for line in f]

rf_model = load_model(r'C:\Users\Mohan\FinalYearProject\SLEEP_DISORDER\FRONTEND\RF_Sleep.pkl')
cnn_model = load_model(r'C:\Users\Mohan\FinalYearProject\SLEEP_DISORDER\FRONTEND\CNN_Sleep.pkl')
lstm_model = load_model(r'C:\Users\Mohan\FinalYearProject\SLEEP_DISORDER\FRONTEND\LSTM_Sleep.pkl')
scaler = load_model(r'C:\Users\Mohan\FinalYearProject\SLEEP_DISORDER\FRONTEND\scaler.pkl')
disorder_mapping = {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}

from django.shortcuts import render, redirect

def home(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')

        try:
            with open('account.txt', 'r') as f:
                accounts = [line.strip().split() for line in f.readlines()]
        except FileNotFoundError:
            return render(request, 'index.html', {'error': 'Account file not found!'})

        for account in accounts:
            if len(account) >= 2 and account[0] == name and account[1] == password:
                # Login successful — redirect to base
                return redirect('base')

        # If no match found
        return render(request, 'index.html', {'error': 'Wrong name or password'})

    return render(request, 'index.html')


def input(request):
    form = SleepForm()
    return render(request, 'input.html', {'form': form})


def output(request):
    if request.method == 'POST':
        form = SleepForm(request.POST)
        if form.is_valid():
            data = np.array([
                form.cleaned_data['gender'],
                form.cleaned_data['age'],
                form.cleaned_data['occupation'],
                form.cleaned_data['sleep_duration'],
                form.cleaned_data['quality_of_sleep'],
                form.cleaned_data['physical_activity_level'],
                form.cleaned_data['stress_level'],
                form.cleaned_data['bmi_category'],
                form.cleaned_data['blood_pressure'],
                form.cleaned_data['heart_rate'],
                form.cleaned_data['daily_steps']
            ]).reshape(1, -1)
            data = scaler.transform(data)

            algo = form.cleaned_data['algorithm']
            if algo == 'RF':
                prediction = disorder_mapping[rf_model.predict(data)[0]]
            elif algo == 'CNN':
                prediction = disorder_mapping[np.argmax(cnn_model.predict(data.reshape(1, 11, 1)))]
            elif algo == 'LSTM':
                prediction = disorder_mapping[np.argmax(lstm_model.predict(data.reshape(1, 11, 1)))]

            return render(request, 'output.html', {'prediction': prediction})
    return render(request, 'input.html', {'form': SleepForm()})

def about(request):
    return render(request, 'about.html')

def team(request):
    return render(request, 'team.html')


def base(request):
    return render(request, 'base.html')


