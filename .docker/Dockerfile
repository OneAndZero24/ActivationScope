# ActivationScope Dockerfile
# Uses manylinux_2_28 base to produce wheels compatible with Python packaging standards.
# PyPI rejects bare linux_x86_64 tags; wheels built here get proper manylinux_2_28_x86_64 tags.

ARG PYTHON_VERSION=3.10
FROM quay.io/pypa/manylinux_2_28_x86_64

# --- Re-declare ALL args after FROM so they're visible inside RUN instructions ---
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.5.1
ARG PLATFORM=cpu

# manylinux is AlmaLinux 8 (RHEL-family) — use dnf, not apt-get.
# Copy target Python binaries into /usr/local/bin so bare "python"/"pip" resolve correctly
# for every RUN instruction and for the CI docker-run wheel-build step.
# POSIX-safe: $(echo ${PYTHON_VERSION} | tr -d '.') turns 3.10 -> 310, producing cp310-cp310 path.
RUN CP="$(printf 'cp%s' "$(echo ${PYTHON_VERSION} | tr -d .)")" && \
    dnf install -y gcc-c++ git && \
    dnf clean all && \
    ln -sf /opt/python/$CP-$CP/bin/pip* /usr/local/bin/ && \
    cp /opt/python/$CP-$CP/bin/python* /usr/local/bin/ && \
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
