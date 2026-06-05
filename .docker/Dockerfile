# ActivationScope Dockerfile

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.5.1
ARG PLATFORM=cpu

FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Generate pyproject.toml in-container (it is git-ignored and built dynamically)
COPY utils/generate-pyproject.py utils/matrix.py ./utils/
COPY matrix.yml pyproject.toml.template setup.py ./
RUN pip install --no-cache-dir pyyaml && python utils/generate-pyproject.py
COPY activationscope/ ./activationscope/
COPY csrc/ ./csrc/

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==${PYTORCH_VERSION} --index-url "https://download.pytorch.org/whl/${PLATFORM}" && \
    pip install .[dev]

COPY tests/ ./tests/

CMD ["pytest", "tests/", "-v", "-x", "--tb=short"]
