FROM python:3.11

ARG POETRY_VERSION="1.5.1"

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN pip3 install --no-cache-dir --upgrade \
    pip \
    virtualenv

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \
    sqlite3 \
    libgeos-dev

RUN apt-get -y clean  \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /root/.cache/clip -p
ADD https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt /root/.cache/clip

RUN mkdir /data && chown appuser: /data -R

USER appuser
WORKDIR /home/appuser

RUN curl -sSL https://install.python-poetry.org |  \
      python - --version ${POETRY_VERSION}

ENV PATH="/home/appuser/.local/bin:$PATH"

COPY ./pyproject.toml ./poetry.lock /home/appuser/

RUN poetry cache clear pypi --all
RUN poetry config virtualenvs.create true  \
    && poetry config virtualenvs.in-project true  \
    && poetry install --no-root --without dev \
    && poetry run poe torch-linux

COPY ./src/ /home/appuser/src/
COPY ./README.md /home/appuser/

RUN poetry install --only-root

RUN poetry run prisma generate --schema=./src/encord_active/lib/db/schema.prisma

WORKDIR /data

RUN git config --global --add safe.directory '*'

ENV HOST=0.0.0.0
EXPOSE 8000

HEALTHCHECK CMD poetry run encord-active --version
ENTRYPOINT ["/home/appuser/.venv/bin/encord-active"]
