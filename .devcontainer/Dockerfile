# syntax=docker/dockerfile:1.3
FROM mcr.microsoft.com/devcontainers/python:1-3.7-bullseye

# Install compiler for cpython
RUN set -ex; \
    apt-get update; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get -yq install build-essential; \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install project in editable mode
RUN --mount=type=bind,target=/tmp/setup-tmp,source=.,rw \
     su ${USERNAME} -c '\
        set -ex; \
        pip install -U pip; \
        pip install -U -e /tmp/setup-tmp/[all]'