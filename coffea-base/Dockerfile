# was: FROM almalinux:8   (or :9)
ARG BASE_IMAGE=almalinux:9
FROM ${BASE_IMAGE}

ARG python
ARG releasev0
ARG TARGETARCH

ENV PYTHON_VERSION=${python}

# Install dependencies for Miniforge
# patch: the pip fastjet==3.4.0.1 sdist build (no prebuilt wheel on
# linux-aarch64) runs `patch` against the vendored FastJet C++ source.
RUN dnf install -y epel-release && \
    dnf update -y && \
    dnf install -y wget bzip2 ca-certificates git wget libgfortran which zsh emacs vim htop man man-pages patch && \
    dnf clean all

# Download and install Miniforge
RUN wget "https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-$(uname)-$(uname -m).sh" -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -bfp /usr/local && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". /usr/local/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". /usr/local/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

COPY  environment.yaml /

# CONDA_OVERRIDE_ARCHSPEC pins the solve to the generic baseline microarch
# for TARGETARCH, so mamba doesn't pick a package build optimized for the
# build host's CPU (e.g. x86_64_v3) that then SIGILLs on other hardware.
RUN export G_SLICE=always-malloc \
    && case "${TARGETARCH}" in \
         amd64) archspec=x86_64 ;; \
         arm64) archspec=aarch64 ;; \
         *) echo "Unsupported TARGETARCH=${TARGETARCH}"; exit 1 ;; \
       esac \
    && mamba install -y python=${PYTHON_VERSION} \
    && CONDA_OVERRIDE_ARCHSPEC=${archspec} mamba env update --file /environment.yaml \
    && mamba clean -tipy \
    && find /usr/local -type f -name '*.a' -delete \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.js.map' -delete \
    && rm -rf /usr/local/pkgs

# Make a symbolic link between installation /opt/conda/etc/grid-security and actual directory /etc/grid-security
RUN ln -s /usr/local/etc/grid-security /etc/grid-security && \
    curl -L https://github.com/opensciencegrid/osg-vo-config/archive/refs/heads/master.tar.gz | \
    tar -xz --strip-components=1 --directory=/etc/grid-security --wildcards */vomses */vomsdir && \
    cp /etc/grid-security/vomses /etc && \
    mv /etc/grid-security/vomses /usr/local/etc/
