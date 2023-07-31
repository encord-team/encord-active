FROM python:3.11

ARG POETRY_VERSION="1.2.2"

WORKDIR /app

COPY .env pyproject.toml poetry.lock README.md /app/
COPY ./src/ /app/src

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg

RUN curl https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt --output ViT-B-32.pt
RUN mkdir /root/.cache/clip -p
COPY ViT-B-32.pt /root/.cache/clip/

RUN apt-get -y clean  \
    && rm -rf /var/lib/apt/lists/*

RUN pip install ".[coco]"

RUN prisma generate --schema=./src/encord_active/lib/db/schema.prisma

WORKDIR /data

RUN git config --global --add safe.directory '*'
EXPOSE 8000

HEALTHCHECK CMD ecord-active --version
ENTRYPOINT ["encord-active"]
