#!/usr/bin/env bash
# Rebuilds wheels/ (pymimir, striplan, pymimir-rgnn, pymimir-rl) inside a manylinux_2_28 container.
#
# Usage, from the repository root on a podman/docker host:
#   podman run --rm -v "$PWD:/code:Z" -w /code \
#       quay.io/pypa/manylinux_2_28_x86_64:2025.11.09-2 bash docker/build_wheels.sh
#
# striplan is built from striplan/. mimir and mimir-rgnn come from src/ if the snapshots
# from the data archive are there, otherwise from GitHub at the pinned commits. Git bundles
# of Loki, nauty and tree-compression in src/ replace their GitHub clones; other third-party
# C++ dependencies are fetched from the network (mimir's CMake, striplan's conan).
set -euo pipefail

MIMIR_REPO=https://github.com/maichmueller/mimir.git
MIMIR_COMMIT=26c62e7eef96ababecb1ccec3b7e5b00bfc0e4d9
RGNN_REPO=https://github.com/simon-stahlberg/mimir-rgnn.git
RGNN_COMMIT=012723aacddc57d8dfe225ceefe9b1eb5df263d8
JOBS=${JOBS:-$(nproc)}

CODE=$(pwd)
WORK=${WORK:-/tmp/beiw-build}
OUT=$CODE/wheels
# SKIP_PYMIMIR=1 reuses an existing pymimir wheel in $OUT.
if [ "${SKIP_PYMIMIR:-0}" = 1 ]; then
    find "$OUT" -name "*.whl" ! -name "pymimir-*" -delete
else
    rm -f "$OUT"/*.whl
fi
export PATH=/opt/python/cp312-cp312/bin:$PATH
export HOME=$WORK/home PIP_CACHE_DIR=$WORK/pip-cache CONAN_HOME=$WORK/conan
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS MAX_JOBS=$JOBS
mkdir -p "$WORK" "$HOME" "$OUT" "$WORK/raw"

# Serve the drexlerd repositories from the bundles, if present: every git clone of their
# GitHub URL (mimir's ExternalProject_Add, striplan's conan recipes) uses a local bare copy.
rm -rf "$WORK/vendor" && mkdir -p "$WORK/vendor"
for b in Loki-02880ea1d nauty-cb44c50bd tree-compression-f487d3fcb; do
    [ -f "$CODE/src/$b.bundle" ] || continue
    repo=${b%-*}
    git clone -q --bare "$CODE/src/$b.bundle" "$WORK/vendor/$repo.git"
    git config --global url."$WORK/vendor/$repo.git".insteadOf "https://github.com/drexlerd/$repo.git"
done
# pymimir and striplan must share a nanobind internals generation (2.7.0 and 2.9.2 are both v16),
# otherwise striplan cannot accept pymimir objects; keep pip-provided build tools on the same line.
printf "nanobind==2.7.0\ncmake<4\n" > "$WORK/constraints.txt"
export PIP_CONSTRAINT=$WORK/constraints.txt

fetch() {  # fetch <dir> <snapshot in src/> <repo> <commit>
    rm -rf "$WORK/$1"
    if [ -f "$CODE/src/$2" ]; then
        tar -xzf "$CODE/src/$2" -C "$WORK"
    else
        git init -q "$WORK/$1"
        git -C "$WORK/$1" fetch -q --depth 1 "$3" "$4"
        git -C "$WORK/$1" checkout -q FETCH_HEAD
    fi
}
fetch mimir mimir-26c62e7ee.tar.gz "$MIMIR_REPO" "$MIMIR_COMMIT"
fetch mimir-rgnn mimir-rgnn-012723a.tar.gz "$RGNN_REPO" "$RGNN_COMMIT"
rm -rf "$WORK/striplan" && cp -r "$CODE/striplan" "$WORK/striplan" && rm -rf "$WORK/striplan/build"
cd "$WORK"

# nauty's configure defaults to -march=native; build it generic so the wheel runs on any x86-64 CPU.
sed -i 's#./configure --prefix=<INSTALL_DIR> CFLAGS=-fPIC#./configure --prefix=<INSTALL_DIR> CFLAGS=-fPIC --enable-generic#' \
    mimir/dependencies/nauty/CMakeLists.txt
grep -q -- '--enable-generic' mimir/dependencies/nauty/CMakeLists.txt

# pymimir: the mimir fork (projective/abstracted IW), built like its cibuildwheel wheels.
if [ "${SKIP_PYMIMIR:-0}" != 1 ]; then
    (cd mimir && pip wheel . --no-deps -w "$WORK/raw" -v)
    auditwheel repair -w "$OUT" "$WORK"/raw/pymimir-*.whl
fi

# striplan: the native backend links libmimir_core.so, which ships inside pymimir.
pip install "$OUT"/pymimir-*.whl "conan>=2.4,<3" "scikit-build-core>=0.10" "cmake<4" ninja
conan profile detect --force
# conancenter's prebuilt b2 needs glibc >= 2.34; manylinux_2_28 has 2.28, so b2 is built from source.
(cd striplan && CMAKE_ARGS="-DCONAN_INSTALL_ARGS=--build=missing;--build=b2/*" STRIPLAN_RPATH_MODE=wheel pip wheel . --no-deps --no-build-isolation -w "$WORK/raw" -v)
auditwheel repair --exclude libmimir_core.so -w "$OUT" "$WORK"/raw/striplan-*.whl

# pymimir-rgnn (pure Python) and pymimir-rl from PyPI, kept as files.
pip wheel ./mimir-rgnn --no-deps -w "$OUT"
pip download "pymimir-rl==0.2.6" --no-deps -d "$OUT"
ls -la "$OUT"
