# Runtime image: the wheels in wheels/ plus pinned PyPI packages; needs no GitHub access.
# build_docker.sh wraps the build; by hand, from the repository root:
#
#   podman build -f docker/Dockerfile -t beiw .
#   podman run --rm -it -v "/path/to/data:/data:Z" beiw
#
# GPU: add --device nvidia.com/gpu=all (podman + CDI) or --gpus all (docker).
# ARM hosts (e.g. Apple Silicon): add --platform linux/amd64 to build and run; the wheels are x86-64.
FROM docker.io/library/python:3.12.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1

# BUILD_TMPDIR lets a host with little disk space keep pip's downloads out of the layers
# (e.g. --build-arg BUILD_TMPDIR=/piptmp -v /big/disk:/piptmp:Z).
ARG BUILD_TMPDIR=/tmp

# torch 2.10.0 and its CUDA libraries (versions from its wheel metadata), largest
# first and split over layers so that no single commit needs much free disk.
RUN TMPDIR=$BUILD_TMPDIR pip install --no-deps torch==2.10.0
RUN TMPDIR=$BUILD_TMPDIR pip install --no-deps nvidia-cudnn-cu12==9.10.2.21
RUN TMPDIR=$BUILD_TMPDIR pip install --no-deps nvidia-cublas-cu12==12.8.4.1
RUN TMPDIR=$BUILD_TMPDIR pip install --no-deps nvidia-nccl-cu12==2.27.5 \
        nvidia-nvshmem-cu12==3.4.5 triton==3.6.0
RUN TMPDIR=$BUILD_TMPDIR pip install --no-deps nvidia-cusparse-cu12==12.5.8.93 \
        nvidia-cusparselt-cu12==0.7.1
RUN TMPDIR=$BUILD_TMPDIR pip install --no-deps nvidia-cusolver-cu12==11.7.3.90 \
        nvidia-cufft-cu12==11.3.3.83 nvidia-curand-cu12==10.3.9.90 \
        nvidia-nvjitlink-cu12==12.8.93 nvidia-cuda-nvrtc-cu12==12.8.93 \
        nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-runtime-cu12==12.8.90 \
        nvidia-nvtx-cu12==12.8.90 nvidia-cufile-cu12==1.13.1.3 \
        cuda-bindings==12.9.4 cuda-pathfinder==1.8.4

# The remaining third-party dependencies, resolved against what is installed above.
COPY docker/requirements.txt /opt/requirements.txt
RUN TMPDIR=$BUILD_TMPDIR pip install -r /opt/requirements.txt && pip check

COPY wheels/ /opt/wheels/
RUN pip install --no-deps /opt/wheels/*.whl \
 && pip check \
 && python -c "import striplan, striplan._iw_encoding_backend, pymimir, pymimir_rgnn, pymimir_rl; print('ok')"

WORKDIR /data
CMD ["bash"]
