FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y \
  build-essential \
  python3.9 \
  python3-pip \
  curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set up environment variables
ENV PATH="$PATH:/root/.poetry/bin" \
  BACKEND_URL="http://backend_api:8080" \
  GRADIO_SERVER_PORT=8080 \
  GRADIO_SERVER_NAME="0.0.0.0"

# Install C++11
RUN apt-get update && apt-get install -y \
  g++ \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.4.0 \
  HNSWLIB_NO_NATIVE=1

# System deps:
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /app
COPY poetry.lock pyproject.toml /app/

# Project initialization:
RUN POETRY_VIRTUALENVS_CREATE=false && \
  poetry install --no-cache --no-interaction --no-ansi --without dev

# Creating folders, and files for a project:
COPY *.py /app/

EXPOSE 8080
CMD HOME=/root \
  poetry run python main.py
