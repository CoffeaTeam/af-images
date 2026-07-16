# Single Dockerfile for every coffea-dask variant. Selected via build args:
#
#   BASE_IMAGE  almalinux:8 | almalinux:9
#   ENV_YAML    environment.yaml | environment-noml.yaml | environment-eaf.yaml
#   NODAK       0 = build ENV_YAML as written
#               1 = strip dask-awkward/dask-histogram and lift the `dask<X` pin
#
# NODAK=1 is only meaningful for a yaml that actually lists dask-awkward
# (currently environment.yaml and environment-eaf.yaml). environment-noml.yaml
# already has neither, and already has no dask pin.
#
# ARGs before FROM are global scope and invisible inside the build stage,
# hence the re-declarations after it.
ARG BASE_IMAGE=almalinux:9
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG python
ARG release
ARG TARGETPLATFORM
ARG ENV_YAML=environment.yaml
ARG NODAK=0

ENV PYTHON_VERSION=${python}

# Install dependencies for Miniforge
RUN dnf install -y epel-release && \
    dnf update -y && \
    dnf install -y wget bzip2 ca-certificates git wget libgfortran which zsh emacs vim htop man man-pages && \
    dnf clean all

RUN wget "https://github.com/conda-forge/miniforge/releases/download/26.1.1-3/Miniforge3-$(uname)-$(uname -m).sh" -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -bfp /usr/local && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    conda clean --force-pkgs-dirs --all --yes && \
    echo ". /usr/local/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". /usr/local/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

COPY *.yaml /tmp/env/

RUN test -f "/tmp/env/${ENV_YAML}" \
      || { echo "ENV_YAML=${ENV_YAML} not found. Available:"; ls -1 /tmp/env/; exit 1; } \
    && cp "/tmp/env/${ENV_YAML}" /environment.yaml

# Single solve: two sequential `mamba env update` passes over an env this size
# (pytorch, mkl, ndcctools; ~400 pkgs / 665MB) gets OOM-killed with exit 137.
#
# NODAK derives its env by filtering rather than from a second yaml, because
# `mamba env update` can only add packages, never remove them.
RUN export G_SLICE=always-malloc \
    && mamba install -y \
        python=${PYTHON_VERSION} \
    && if [ "$NODAK" = "1" ]; then \
         grep -qE '^[[:space:]]*-[[:space:]]*dask-awkward\b' /environment.yaml \
           || { echo "NODAK=1 but ${ENV_YAML} does not list dask-awkward"; exit 1; }; \
         # Drop dask-awkward/dask-histogram, and rewrite `- dask<X` to `- dask`.
         # The `<` right after `dask` is what keeps the sed off dask-jobqueue,
         # dask-gateway, dask-core and dask_labextension.
         grep -vE '^[[:space:]]*-[[:space:]]*dask-(awkward|histogram)\b' /environment.yaml \
           | sed -E 's/^([[:space:]]*-[[:space:]]*dask)[[:space:]]*<[^[:space:]]+[[:space:]]*$/\1/' \
           > /env-nodak.yaml; \
         ENV_FILE=/env-nodak.yaml; \
       else \
         ENV_FILE=/environment.yaml; \
       fi \
    # Record the capability the yaml ASKS FOR, so the check below compares the
    # yaml against the solver -- the drift that actually matters. Deriving it
    # here rather than from a build arg means the workflow and the yaml can
    # never disagree. Both packages are required: hist.dask needs
    # dask_histogram, and dask-awkward alone does not provide it.
    && if grep -qE '^[[:space:]]*-[[:space:]]*dask-awkward\b' ${ENV_FILE} \
       && grep -qE '^[[:space:]]*-[[:space:]]*dask-histogram\b' ${ENV_FILE}; then \
         echo 1 > /etc/af_dask_awkward; \
       else \
         echo 0 > /etc/af_dask_awkward; \
       fi \
    && echo "=== base=${BASE_IMAGE} yaml=${ENV_YAML} nodak=${NODAK} solving ${ENV_FILE}" \
    && echo "=== declared dask-awkward capability: $(cat /etc/af_dask_awkward)" \
    && echo "=== dask specs:" \
    && grep -nE '^[[:space:]]*-[[:space:]]*dask' ${ENV_FILE} \
    && CONDA_OVERRIDE_ARCHSPEC=x86_64 mamba env update --file ${ENV_FILE} \
    && rm -f /env-nodak.yaml \
    && rm -rf /tmp/env \
    && mamba clean -tipy \
    && find /usr/local -type f -name '*.a' -delete \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.js.map' -delete \
    && rm -rf /usr/local/pkgs

# Fail the build if the solver disagreed with the yaml. Without this, a
# transitive dep dragging dask-awkward back in (or a solve quietly dropping it)
# produces a green build that tested the wrong thing.
RUN python - <<'PY'
import importlib.util, sys
want = open("/etc/af_dask_awkward").read().strip() == "1"
have = all(importlib.util.find_spec(m) for m in ("dask_awkward", "dask_histogram"))
if want != have:
    sys.exit(f"yaml declared dask-awkward={want} but installed env has {have}")
print(f"variant OK: dask-awkward capability = {have}")
PY

# The readout: diff these between coffea-dak-almalinux9 and coffea-almalinux9.
RUN python -c "import dask, distributed; print('dask', dask.__version__, '| distributed', distributed.__version__)"

# Make a symbolic link between installation /opt/conda/etc/grid-security and actual directory /etc/grid-security
RUN ln -s /usr/local/etc/grid-security /etc/grid-security && \
    curl -L https://github.com/opensciencegrid/osg-vo-config/archive/refs/heads/master.tar.gz | \
    tar -xz --strip-components=1 --directory=/etc/grid-security --wildcards */vomses */vomsdir && \
    cp /etc/grid-security/vomses /etc && \
    mv /etc/grid-security/vomses /usr/local/etc/