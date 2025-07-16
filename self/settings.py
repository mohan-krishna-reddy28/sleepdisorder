# FRONTEND/self/settings.py
"""
Django settings for webapp project.
"""

import os
import dj_database_url

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'your-fallback-insecure-key-please-change-in-prod')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DJANGO_DEBUG', 'False') == 'True'

ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
if not DEBUG:
    RENDER_EXTERNAL_HOSTNAME = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
    if RENDER_EXTERNAL_HOSTNAME:
        ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)
    # Add any custom domains here if you have them, e.g.:
    # ALLOWED_HOSTS.append('yourcustomdomain.com')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'webapp', # Your Django app
]

MIDDLEWARE = [
    'whitenoise.middleware.WhiteNoiseMiddleware', # Keep this at the top if serving static via WhiteNoise
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'self.urls' # Points to your main project's urls.py

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')], # Project-wide templates, if any (e.g., base.html for login)
        'APP_DIRS': True, # This enables Django to find templates within each app's 'templates' folder (e.g., webapp/templates/)
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'self.wsgi.application'


# Database
DATABASES = {
    'default': dj_database_url.config(
        default=os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3'),
        conn_max_age=600
    )
}

# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators
# (The rest of your password validators go here)


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/
# (The rest of your i18n settings go here)


# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'

# IMPORTANT: Ensure this list correctly points to ALL your static file directories.
# Keep only the relevant paths. If you only have static files in webapp/static/,
# remove the os.path.join(BASE_DIR, 'static') line.
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),         # For project-level static files (e.g., FRONTEND/static/)
    os.path.join(BASE_DIR, 'webapp', 'static'), # For app-level static files (e.g., FRONTEND/webapp/static/)
]

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles') # Where `collectstatic` will gather all static files

# Corrected STATICFILES_STORAGE for WhiteNoise
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'