FROM python:3.9.16-slim-bullseye

# Install C++11
RUN apt-get update && apt-get install -y \
  g++ \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV HOME=/root \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.4.0

# System deps:
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
WORKDIR /app
COPY poetry.lock pyproject.toml /app/

# Project initialization:
RUN POETRY_VIRTUALENVS_CREATE=false && poetry install --no-interaction --no-ansi

COPY *.py /app/

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "main:app", "--port", "8000"]
