from pathlib import Path
import os
import dj_database_url

# 1) BASE_DIR
BASE_DIR = Path(__file__).resolve().parent.parent

# 2) SECRET_KEY y DEBUG desde entorno
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')
if not SECRET_KEY:
    raise RuntimeError("Define la variable de entorno DJANGO_SECRET_KEY")

DEBUG = os.getenv('DJANGO_DEBUG', 'False') == 'True'

# 3) Hosts permitidos
ALLOWED_HOSTS = [
    '.railway.app',
    '.railway.internal',
    'proyecto-prophet-tqqq-production.up.railway.app',
    'localhost',
    '127.0.0.1',
]

# 4) APPS instaladas
INSTALLED_APPS = [
    # Tus apps
    'accounts.apps.AccountsConfig',
    'dashboard.apps.DashboardConfig',

    # Django core
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

# 5) MIDDLEWARE
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # sirve archivos estáticos
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# 6) URLs y WSGI
ROOT_URLCONF = 'proyecto3.urls'
WSGI_APPLICATION = 'proyecto3.wsgi.application'

# 7) Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
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

# 8) Base de datos: usa PostgreSQL en producción, SQLite en local
DATABASES = {
    'default': dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600,
    )
}

# 9) Validadores de contraseña
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# 10) Internacionalización
LANGUAGE_CODE = 'es'
TIME_ZONE = 'America/Costa_Rica'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# 11) Archivos estáticos
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# 12) Redirecciones tras login/logout
LOGIN_REDIRECT_URL = 'elegir_dashboard'
LOGOUT_REDIRECT_URL = 'login'

# 13) Configuración de email (solo consola en dev)
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
DEFAULT_FROM_EMAIL = 'webmaster@localhost'

# 14) Auto field por defecto
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# 15) Seguridad para producción (CSRF y cookies sobre HTTPS)
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True
CSRF_FAILURE_VIEW = 'django.views.csrf.csrf_failure'
