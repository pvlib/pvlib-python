# syntax=docker/dockerfile:1.3
FROM mcr.microsoft.com/devcontainers/python:1-3.7-bullseye

# Install compiler for cpython
RUN set -ex; \
    apt-get update; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get -yq install build-essential; \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*