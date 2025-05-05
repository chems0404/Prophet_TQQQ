# Usa una imagen base de Python
FROM python:3.11-slim

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

# Comando para ejecutar la aplicación (ajústalo según tu app)
CMD ["python", "app.py"]
