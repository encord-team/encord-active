FROM python:3.9-slim

ARG POETRY_VERSION="1.2.2"

WORKDIR /app

COPY .env pyproject.toml poetry.lock README.md /app
COPY ./src/ /app/src

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg

RUN apt-get -y clean  \
    && rm -rf /var/lib/apt/lists/*

RUN pip install .

WORKDIR /data

RUN git config --global --add safe.directory '*'
EXPOSE 8501
EXPOSE 8000

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

HEALTHCHECK CMD ecord-active --version
ENTRYPOINT ["encord-active"]

