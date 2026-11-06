#!/usr/bin/env bash
# Builds an Apptainer image from docker/beiw.def, for hosts without docker/podman.
#
#   ./build_apptainer.sh [image.sif]     # default: beiw.sif next to this script
#
# The build unpacks ~20 GB, so APPTAINER_TMPDIR (and APPTAINER_CACHEDIR) default to
# /var/tmp/$USER/apptainer; point them at another local disk if /var/tmp is small.
# Network filesystems often fail as build scratch. Non-root users build with --fakeroot.
set -euo pipefail

SIF=${1:-$(cd "$(dirname "$0")" && pwd)/beiw.sif}
case $SIF in /*) ;; *) SIF=$PWD/$SIF ;; esac
cd "$(dirname "$0")"

APPTAINER=${APPTAINER:-$(command -v apptainer || command -v singularity || true)}
if [ -z "$APPTAINER" ]; then
    echo "error: apptainer (or singularity) not found" >&2
    exit 1
fi
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-/var/tmp/$USER/apptainer}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$APPTAINER_TMPDIR/cache}
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

if [ "$(id -u)" = 0 ]; then FAKEROOT=(); else FAKEROOT=(--fakeroot); fi
"$APPTAINER" build "${FAKEROOT[@]}" "$SIF" docker/beiw.def

cat <<EOF

Built $SIF. Run the planner from the data folder (the one holding datasets/ and checkpoints/):
  $(basename "$APPTAINER") exec --nv --bind "\$PWD:/data" --pwd /data $SIF python -m striplan.plan ...
Drop --nv (and pass --cpu to the planner) on machines without a GPU.
EOF
