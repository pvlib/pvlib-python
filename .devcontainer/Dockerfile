FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Install compiler for cpython
RUN set -ex; \
    apt-get update; \
    export DEBIAN_FRONTEND=noninteractive; \
    apt-get -yq install build-essential; \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/conda-tmp/

# Install base dev utilities
# Uninstall pylint as is activated by default in vscode
# Use mamba solver to reduce startup time
RUN set -ex; \
    pipx uninstall pylint || true; \
    su vscode -c '\
    /opt/conda/bin/conda install -n base conda-libmamba-solver; \ 
    /opt/conda/bin/conda config --set solver libmamba; \
    /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml'; \
    rm -rf /tmp/conda-tmp
