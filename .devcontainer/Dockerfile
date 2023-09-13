# syntax=docker/dockerfile:1.3
FROM mcr.microsoft.com/devcontainers/python:1-3.7-bullseye

# Install compiler for Cython
RUN set -ex; \
    apt-get update; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get -yq install build-essential; \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Dependencies in Advance
# Pre-installing these packages improves the startup time for Codespaces instances when using prebuilds.
# See GitHub Docs for more info: https://docs.github.com/en/codespaces/prebuilding-your-codespaces
RUN --mount=type=bind,target=/tmp/setup-tmp,source=.,rw \
     su ${USERNAME} -c '\
        set -ex; \
        pip install -U pip; \
        pip install -U -e /tmp/setup-tmp/[all]'
