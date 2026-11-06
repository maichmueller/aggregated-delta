#!/usr/bin/env bash
# Builds the runtime image from docker/Dockerfile with docker or podman.
#
#   ./build_docker.sh [tag]          # default tag: beiw
#
# ENGINE=podman (or docker) picks the engine; otherwise docker is used if installed, else podman.
# The wheels are x86-64, so the image is always built for linux/amd64; on ARM hosts
# (e.g. Apple Silicon) it then runs under emulation.
set -euo pipefail
cd "$(dirname "$0")"

TAG=${1:-beiw}
ENGINE=${ENGINE:-$(command -v docker || command -v podman || true)}
if [ -z "$ENGINE" ]; then
    echo "error: neither docker nor podman found (set ENGINE=...)" >&2
    exit 1
fi
NAME=$(basename "$ENGINE")

"$ENGINE" build --platform linux/amd64 -f docker/Dockerfile -t "$TAG" .

if [ "$NAME" = podman ]; then GPU="--device nvidia.com/gpu=all"; VOL=":Z"; else GPU="--gpus all"; VOL=""; fi
cat <<EOF

Built image '$TAG'. Run it from the data folder (the one holding datasets/ and checkpoints/):
  $NAME run --rm -it --platform linux/amd64 -v "\$PWD:/data$VOL" $TAG
Add '$GPU' for GPUs (NVIDIA container toolkit required); planning without one needs --cpu.
EOF
