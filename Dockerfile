# Base leve do Python
FROM python:3.11-slim

# Evita prompts e instala dependências do sistema para OpenCV
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Define a pasta de trabalho
WORKDIR /app

# Copia o projeto
COPY . /app

# Instala as dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn \
        opencv-python-headless \
        numpy \
        requests \
        python-dotenv \
        Pillow \
        python-multipart

# Expõe a porta da API
EXPOSE 5000

# Comando para iniciar a aplicação
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
