FROM python:3.9-slim

ARG POETRY_VERSION="1.2.2"

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
    sqlite3

RUN apt-get -y clean  \
    && rm -rf /var/lib/apt/lists/*


USER appuser
WORKDIR /home/appuser

RUN curl -sSL https://install.python-poetry.org |  \
      python - --version ${POETRY_VERSION}

ENV PATH="/home/appuser/.local/bin:$PATH"

COPY .env pyproject.toml poetry.lock /home/appuser/

RUN poetry cache clear pypi --all
RUN poetry config virtualenvs.create true  \
    && poetry config virtualenvs.in-project true  \
    && poetry config experimental.new-installer true  \
    && poetry install --no-root --without dev

COPY ./src/ /home/appuser/src
COPY README.md /home/appuser

RUN poetry install --only-root

EXPOSE 8501

COPY run.sh /home/appuser

USER root
RUN mkdir /data && chown appuser: /data -R
USER appuser

RUN git config --global --add safe.directory '*'
EXPOSE 8000

ENTRYPOINT ["./run.sh"]
