# was: FROM almalinux:8   (or :9)
ARG BASE_IMAGE=almalinux:9
FROM ${BASE_IMAGE}

ARG python
ARG releasev0
ARG TARGETARCH

ENV PYTHON_VERSION=${python}

# Install dependencies for Miniforge
RUN dnf install -y epel-release && \
    dnf update -y && \
    dnf install -y wget bzip2 ca-certificates git wget libgfortran which zsh emacs vim htop man man-pages && \
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

# Python TLS clients (aiohttp/fsspec HTTP, requests, urllib) trust only the
# Mozilla bundle from conda's ca-certificates, not the IGTF grid CAs that
# ca-policy-lcg installs in /etc/grid-security/certificates. Since Nov 2026,
# xrootd-local.unl.edu:1094 (AGC inputs over HTTPS) presents a chain ending in
# the IGTF-only root "emSign Root TLS CA - G1", so those reads fail with
# CERTIFICATE_VERIFY_FAILED. Add just that one root (fingerprint-pinned) to a
# copy of the Mozilla bundle and point OpenSSL (SSL_CERT_FILE) and requests
# (REQUESTS_CA_BUNDLE) at it. To trust another grid root, add it the same way.
ARG EXTRA_ROOT=/etc/grid-security/certificates/emSignRootTLSCAG1.pem
ARG EXTRA_ROOT_SHA256=CE:F7:1E:70:B7:C2:9A:DD:F6:C3:0C:D1:9E:61:4B:38:FD:5F:02:A4:35:A0:EE:DD:D0:08:7E:18:3D:10:1A:51
RUN mozilla="$(python -c 'import ssl; print(ssl.get_default_verify_paths().openssl_cafile)')" \
    && test -s "$mozilla" \
    && fp="$(openssl x509 -in "$EXTRA_ROOT" -noout -fingerprint -sha256 | cut -d= -f2)" \
    && { [ "$fp" = "$EXTRA_ROOT_SHA256" ] || { echo "unexpected fingerprint for $EXTRA_ROOT: $fp"; exit 1; }; } \
    && cat "$mozilla" "$EXTRA_ROOT" > /usr/local/etc/af-ca-bundle.pem \
    && openssl verify -CAfile /usr/local/etc/af-ca-bundle.pem "$EXTRA_ROOT"

ENV SSL_CERT_FILE=/usr/local/etc/af-ca-bundle.pem \
    REQUESTS_CA_BUNDLE=/usr/local/etc/af-ca-bundle.pem
