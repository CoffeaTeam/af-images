FROM condaforge/mambaforge:23.3.1-1

ARG python
ARG release

ENV PYTHON_VERSION=${python}

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="Etc/UTC"

RUN apt-get update \
     && apt-get install -yq --no-install-recommends libarchive-dev zsh \
     && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY environment.yml /

RUN mamba install --yes python=${PYTHON_VERSION} \
     && mamba env update --file /environment.yml \
     && mamba clean -y --all

# Make a symbolic link between installation /opt/conda/etc/grid-security and actual directory /etc/grid-security
RUN ln -s /opt/conda/etc/grid-security /etc/grid-security
