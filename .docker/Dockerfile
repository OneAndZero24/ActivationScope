# ActivationScope Dockerfile
# Uses manylinux_2_28 base to produce wheels compatible with Python packaging standards.
# PyPI rejects bare linux_x86_64 tags; wheels built here get proper manylinux_2_28_x86_64 tags.

ARG PYTHON_VERSION=3.10
FROM quay.io/pypa/manylinux_2_28_x86_64

# Re-declare args so they are visible inside the build stage.
ARG PYTORCH_VERSION=2.5.1
ARG PLATFORM=cpu

# manylinux is AlmaLinux 8 (RHEL-family) — use dnf, not apt-get.
# Copy target Python and pip into /usr/local/bin so bare "python" / "pip" resolve correctly
# for every subsequent RUN and for the CI wheel-build step on docker run.
RUN dnf install -y gcc-c++ git && \
    dnf clean all && \
    PY="$(echo $PYTHON_VERSION | sed 's/3\.\([0-9]\)/cp3\1-cp3\1/')" && \
    cp /opt/python/$PY/bin/pip /usr/local/bin/ && \
    cp /opt/python/$PY/bin/python* /usr/local/bin/ && \
    pip install --upgrade pip setuptools wheel

WORKDIR /src

# Generate pyproject.toml in-container (it is git-ignored and built dynamically)
COPY utils/generate-pyproject.py utils/matrix.py ./utils/
COPY matrix.yml pyproject.toml.template setup.py ./
RUN pip install --no-cache-dir pyyaml && python utils/generate-pyproject.py

COPY activationscope/ ./activationscope/
COPY csrc/ ./csrc/

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==${PYTORCH_VERSION} --index-url "https://download.pytorch.org/whl/${PLATFORM}" && \
    pip install --no-build-isolation .[dev]

COPY tests/ ./tests/

CMD ["pytest", "tests/", "-v", "-x", "--tb=short"]
