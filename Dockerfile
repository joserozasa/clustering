ARG PROJECT_DIRECTORY="/usr/src/app"
ARG PYTHON_VER=3.9


FROM python:${PYTHON_VER}-slim-buster AS build-env

# Instalar paquetes necesarios
RUN apt-get update \
    && apt-get install -y libopencv-dev \
    && apt-get install -y git \
    && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copiar código fuente
COPY crop_images.py .
COPY crop_images_cpu.py .
COPY clustering.py .
COPY create_catalog.py .


# Instalar dependencias del proyecto
COPY requirements.txt .
RUN pip install -r requirements.txt

# Establecer entrada por defecto
ENTRYPOINT ["python", "create_catalog.py"]
