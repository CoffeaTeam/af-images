# Single Dockerfile for all distro variants. Selected via build args:
#
#   almalinux8       --build-arg BASE_IMAGE=almalinux:8 --build-arg ENV_YAML=environment.yaml
#   almalinux9       --build-arg BASE_IMAGE=almalinux:9 --build-arg ENV_YAML=environment.yaml
#   almalinux9-noml  --build-arg BASE_IMAGE=almalinux:9 --build-arg ENV_YAML=environment-noml.yaml
#   almalinux9-eaf   --build-arg BASE_IMAGE=almalinux:9 --build-arg ENV_YAML=environment-eaf.yaml
#
# ARGs declared before FROM are in the global scope and are NOT visible inside
# the build stage -- hence the re-declarations after FROM.
ARG BASE_IMAGE=almalinux:9
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG python
ARG release
ARG TARGETPLATFORM
# Which environment.yaml to build from (varies by distro flavour).
ARG ENV_YAML=environment.yaml
# 1 -> coffea-dak-x : with dask-awkward/dask-histogram, dask pin as written
# 0 -> coffea-x     : dask-awkward/dask-histogram filtered out, dask pin lifted
ARG WITH_DASK_AWKWARD=1

ENV PYTHON_VERSION=${python}

# Install dependencies for Miniforge.
# Package names below are common to AlmaLinux 8 and 9; if a variant ever needs
# a different set, that is the point to split this file again.
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

# Copy every env file, then select one. Copying the lot (rather than
# `COPY ${ENV_YAML} /`) keeps the layer cache key identical across flavours
# that share a yaml, and fails loudly below if the name is wrong.
COPY *.yaml /tmp/env/

RUN test -f "/tmp/env/${ENV_YAML}" \
      || { echo "ENV_YAML=${ENV_YAML} not found. Available:"; ls -1 /tmp/env/; exit 1; } \
    && cp "/tmp/env/${ENV_YAML}" /environment.yaml

# Single solve. Two sequential `mamba env update` passes over an env this size
# (pytorch, mkl, ndcctools; ~400 pkgs / 665MB) gets OOM-killed with exit 137.
#
# The nodak variant is derived by filtering rather than by a second yaml,
# because `mamba env update` can only add packages, never remove them.
RUN export G_SLICE=always-malloc \
    && mamba install -y \
        python=${PYTHON_VERSION} \
    && if [ "$WITH_DASK_AWKWARD" = "1" ]; then \
         ENV_FILE=/environment.yaml; \
       else \
         grep -qE '^[[:space:]]*-[[:space:]]*dask-(awkward|histogram)\b' /environment.yaml \
           || { echo "expected dask-awkward/dask-histogram in ${ENV_YAML}"; exit 1; }; \
         # Drop dask-awkward/dask-histogram, and rewrite `- dask<X` to `- dask`.
         # The `<` right after `dask` is what stops the sed touching
         # dask-jobqueue, dask-gateway, dask-core or dask_labextension.
         grep -vE '^[[:space:]]*-[[:space:]]*dask-(awkward|histogram)\b' /environment.yaml \
           | sed -E 's/^([[:space:]]*-[[:space:]]*dask)[[:space:]]*<[^[:space:]]+[[:space:]]*$/\1/' \
           > /env-nodak.yaml; \
         ENV_FILE=/env-nodak.yaml; \
       fi \
    && echo "=== base=${BASE_IMAGE} yaml=${ENV_YAML} solving ${ENV_FILE}, dask specs: ===" \
    && grep -nE '^[[:space:]]*-[[:space:]]*dask' ${ENV_FILE} \
    && CONDA_OVERRIDE_ARCHSPEC=x86_64 mamba env update --file ${ENV_FILE} \
    && rm -f /env-nodak.yaml \
    && rm -rf /tmp/env \
    && mamba clean -tipy \
    && find /usr/local -type f -name '*.a' -delete \
    && find /usr/local -type f -name '*.pyc' -delete \
    && find /usr/local -type f -name '*.js.map' -delete \
    && rm -rf /usr/local/pkgs

# Bake the declared capability into the image so the tests can read it.
# After the env layer, so flipping the variant does not invalidate the
# dnf/miniforge layers above.
ENV AF_DASK_AWKWARD=${WITH_DASK_AWKWARD}

# Fail the build if the solver did not do what we asked. Without this, a
# regression that drags dask-awkward back in produces a green build that
# tested the wrong thing.
RUN python - <<'PY'
import importlib.util, os, sys
want = os.environ["AF_DASK_AWKWARD"] == "1"
have = all(importlib.util.find_spec(m) for m in ("dask_awkward", "dask_histogram"))
if want != have:
    sys.exit(f"AF_DASK_AWKWARD={os.environ['AF_DASK_AWKWARD']} but dask-awkward present={have}")
print(f"variant OK: dask-awkward present={have}")
PY

# The experiment readout: diff these two lines between the dak and nodak builds.
RUN python -c "import dask, distributed; print('dask', dask.__version__, '| distributed', distributed.__version__)"

# Make a symbolic link between installation /opt/conda/etc/grid-security and actual directory /etc/grid-security
RUN ln -s /usr/local/etc/grid-security /etc/grid-security && \
    curl -L https://github.com/opensciencegrid/osg-vo-config/archive/refs/heads/master.tar.gz | \
    tar -xz --strip-components=1 --directory=/etc/grid-security --wildcards */vomses */vomsdir && \
    cp /etc/grid-security/vomses /etc && \
    mv /etc/grid-security/vomses /usr/local/etc/