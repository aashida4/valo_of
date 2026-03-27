FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_VERSION=4.9.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-numpy \
    cmake build-essential pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    git wget ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy

# Build OpenCV with CUDA support
RUN cd /tmp && \
    wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -O opencv.tar.gz && \
    wget -q https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz -O opencv_contrib.tar.gz && \
    tar xzf opencv.tar.gz && tar xzf opencv_contrib.tar.gz && \
    mkdir opencv-${OPENCV_VERSION}/build && cd opencv-${OPENCV_VERSION}/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN="7.5;8.0;8.6;8.9;9.0" \
        -DWITH_CUDNN=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -DPYTHON3_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
        -DBUILD_opencv_python3=ON \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_opencv_cudaoptflow=ON \
        -DBUILD_opencv_cudaimgproc=ON \
        -DBUILD_opencv_cudaarithm=ON \
        -DBUILD_opencv_cudalegacy=ON && \
    make -j$(nproc) && make install && ldconfig && \
    rm -rf /tmp/opencv* /tmp/opencv_contrib*

WORKDIR /app
COPY optical_flow.py /app/

ENTRYPOINT ["python3", "optical_flow.py"]
