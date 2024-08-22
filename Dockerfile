# Build arguments 
# This is a relatively large image, so we might want to use a smaller base image, such as
# alpine in the future if image size becomes an issue.
ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE} as base


ARG POETRY_VERSION=1.8.2
ARG PYTHON_VERSION=3.11.6
ARG POETRY_HOME="/opt/poetry"
ARG ROOT_DIR="/root"
ARG FLATNAV_PATH="${ROOT_DIR}/flatnavlib"
ARG INCLUDE_HNSWLIB=false


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        # Need for python installation: 
        # https://github.com/pyenv/pyenv/wiki#suggested-build-environment
        make \
        build-essential \
        ca-certificates \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        # Multi-process manager inside docker 
        supervisor \
        # Install the rest
        git \
        gcc \
        g++ \
        apt-utils \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Install python 
# We use pyenv to manage python versions 
ENV PYENV_ROOT=$HOME/.pyenv

# Shims are small proxy executables that intercept calls to Python commands. 
# Putting $PYENV_ROOT/shims at the beginning of PATH ensures that the shimmed 
# Python commands are found and used before any system-wide Python installations.
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH 

ENV PYTHON_VERSION=${PYTHON_VERSION}


RUN set -ex \
    && curl -L https://pyenv.run | /bin/sh \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash 

# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME=${POETRY_HOME} \
    POETRY_VERSION=${POETRY_VERSION} 

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - 

# Add poetry to PATH
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR ${FLATNAV_PATH}

# Copy source code
COPY flatnav/ ./flatnav/
COPY flatnav_python/ ./flatnav_python/
COPY experiments/ ./experiments/
# We don't need quantization for now, but it doesn't hurt to have it.
COPY quantization/ ./quantization/
# Copy external dependencies (for now only cereal)
COPY external/ ./external/

# Copy the configuration for supervisor 
COPY bin/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install needed dependencies including flatnav. 
# This installs numpy as well, which is a large dependency. 
WORKDIR ${FLATNAV_PATH}/flatnav_python
RUN ./install_flatnav.sh 

WORKDIR ${FLATNAV_PATH}
# Install hnswlib (from a forked repo that has extensions we need)
RUN if [ "$INCLUDE_HNSWLIB" = true ] ; then \
        git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git \
        && cd hnswlib-original/python_bindings \
        && poetry install --no-root \
        && poetry run python setup.py bdist_wheel; \
    fi 

# Get the wheel as an environment variable 
# NOTE: This is not robust and will break if there are multiple wheels in the dist folder
ENV FLATNAV_WHEEL=${FLATNAV_PATH}/flatnav_python/dist/*.whl

WORKDIR ${FLATNAV_PATH}/experiments
RUN rm poetry.lock
RUN echo ${FLATNAV_WHEEL}
RUN poetry add ${FLATNAV_WHEEL} -v

RUN if [ "$INCLUDE_HNSWLIB" = true ] ; then \
        export HNSWLIB_WHEEL=${FLATNAV_PATH}/hnswlib-original/python_bindings/dist/*.whl; \
        poetry add ${HNSWLIB_WHEEL}; \
    fi

RUN poetry install --no-root
