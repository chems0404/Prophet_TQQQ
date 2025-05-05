# Usa una imagen base de Python
FROM python:3.11-slim

# Instalar herramientas de compilación necesarias
RUN apt-get update && apt-get install -y build-essential gcc g++

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de requerimientos dentro del contenedor
COPY requirements.txt .

# Instala las dependencias especificadas en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu aplicación al contenedor
COPY . .

# Expone el puerto en el que tu aplicación va a correr
EXPOSE 8000

# Ejecutar las migraciones de base de datos y luego Gunicorn
CMD ["sh", "-c", "python manage.py migrate && gunicorn proyecto3.wsgi:application --bind 0.0.0.0:8000"]
