#!/bin/bash
# fastjet has no prebuilt wheel for linux-aarch64, so `pip install fastjet`
# builds it from source (./autogen.sh + make) against the bundled CGAL. Two
# things break that build with conda-forge's aarch64 gcc/binutils toolchain:
#
# 1. libtool mis-detects "postdeps": it parses `$CXX -v conftest.cpp -o x`'s
#    default library search dirs, which this toolchain prints as two-token
#    "-L <dir>" (space-separated) rather than the one-token "-L<dir>" style
#    libtool's regex expects. It strips the directory but leaves a dangling,
#    argument-less "-L" at the end of `postdeps`. At link time that dangling
#    "-L" swallows crtendS.o as its argument instead of linking it in, so
#    __TMC_END__ (referenced by register_tm_clones/deregister_tm_clones in
#    crtbeginS.o) never gets defined:
#      ld: hidden symbol `__TMC_END__' isn't defined
#    Fix: after autogen.sh generates the `libtool` scripts (there is one per
#    subdirectory with its own configure), strip the dangling "-L" from
#    postdeps/predeps before running make.
#
# 2. The vendored D0RunIICone plugin has a latent bug (a template member
#    access that isn't valid under strict two-phase lookup) that GCC 13+'s
#    -Wtemplate-body diagnostic promotes to a hard error. Harmless dead code
#    otherwise, so just disable that one warning.
#
# Once `src/fastjet/_fastjet_core` is already populated, fastjet's own
# setup.py skips its internal CGAL/autogen/make build entirely (see
# FastJetBuild.build_extensions: `if not OUTPUT.exists(): ...`) and only
# compiles the thin pybind11 wrapper against it -- so building it here
# ourselves, then handing the same directory to `pip install`, is enough to
# get a working install without patching fastjet's own setup.py.
set -euo pipefail

FASTJET_VERSION="${1:?usage: build_fastjet_arm64.sh <fastjet-version>}"
CGAL_VERSION="5.5.1"

WORKDIR=$(mktemp -d)
trap 'rm -rf "${WORKDIR}"' EXIT
cd "${WORKDIR}"

pip download --no-binary fastjet --no-deps -d . "fastjet==${FASTJET_VERSION}"
tar xzf "fastjet-${FASTJET_VERSION}.tar.gz"
cd "fastjet-${FASTJET_VERSION}"

curl -sL -o cgal.zip \
  "https://github.com/CGAL/cgal/releases/download/v${CGAL_VERSION}/CGAL-${CGAL_VERSION}-library.zip"
python3 -c "import zipfile; zipfile.ZipFile('cgal.zip').extractall('.')"

cd fastjet-core
patch -p0 --forward pyinterface/fastjet.i ../patch_fastjet_i.txt
patch -p0 --forward src/ClusterSequence.cc ../patch_clustersequence.txt

export PYTHON="$(command -v python3)"
export PYTHON_INCLUDE="-I$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
export CXXFLAGS="-O3 -Bstatic -lgmp -Bdynamic -Wno-template-body"
export ORIGIN='$ORIGIN'

./autogen.sh \
  --prefix="${WORKDIR}/fastjet-${FASTJET_VERSION}/src/fastjet/_fastjet_core" \
  --enable-allcxxplugins --enable-cgal-header-only --enable-cgal \
  --with-cgaldir="${WORKDIR}/fastjet-${FASTJET_VERSION}/CGAL-${CGAL_VERSION}" \
  --enable-swig --enable-pyext \
  'LDFLAGS=-Wl,-rpath=$$ORIGIN/_fastjet_core/lib:$$ORIGIN'

# See point 1 above. There is one generated `libtool` script per
# subdirectory (e.g. plugins/SISCone/siscone/libtool) -- patch them all.
find . -name libtool | while read -r lt; do
  while grep -qE '^(postdeps|predeps)="(.*) -L"$' "${lt}"; do
    sed -i -E 's/^(postdeps|predeps)="(.*) -L"$/\1="\2"/' "${lt}"
  done
done

# CGAL's header-only templates make each cc1plus fairly memory-hungry;
# capping jobs (rather than a bare/nproc-wide `-j`) keeps peak memory use
# predictable regardless of how many cores the runner has.
jobs=$(nproc); [ "${jobs}" -gt 4 ] && jobs=4
make -j"${jobs}"
make install

cd "${WORKDIR}/fastjet-${FASTJET_VERSION}"
pip install .
