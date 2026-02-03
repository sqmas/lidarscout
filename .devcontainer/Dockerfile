FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y wget ca-certificates git libgl1-mesa-dev libglu1-mesa-dev libglew-dev libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev libaio-dev gcc-14 g++-14 ninja-build gdb
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-14
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.11/cmake-3.31.11-linux-x86_64.sh -O cmake.sh
RUN mkdir cmake && sh cmake.sh --skip-license --exclude-subdir --prefix=/cmake
ENV PATH="${PATH}:/cmake/bin"
ENV CUDA_PATH="/usr/local/cuda-12.8/targets/x86_64-linux"
USER ubuntu
