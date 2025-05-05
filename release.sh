#!/bin/bash

echo "▶ Ejecutando migraciones..."
python manage.py migrate

echo "▶ Ejecutando collectstatic..."
python manage.py collectstatic --noinput

echo "▶ Creando superusuario (si no existe)..."
python manage.py shell << END
import os
from django.contrib.auth import get_user_model

username = os.environ.get("ADMIN_USER")
email = os.environ.get("ADMIN_EMAIL")
password = os.environ.get("ADMIN_PASS")

User = get_user_model()
if username and email and password:
    if not User.objects.filter(username=username).exists():
        User.objects.create_superuser(username, email, password)
        print(f"✅ Superusuario '{username}' creado.")
    else:
        print(f"ℹ️ El usuario '{username}' ya existe.")
else:
    print("⚠️ Variables de entorno ADMIN_USER, ADMIN_EMAIL o ADMIN_PASS no definidas.")
END
