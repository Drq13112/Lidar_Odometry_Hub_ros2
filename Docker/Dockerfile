# Base CUDA 12.2 con Ubuntu 22.04
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04


ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'
RUN apt-get update

# Actualizar e instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    build-essential \
    cmake \
    git \
    locales \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*


RUN groupadd --gid $GROUP_ID david && \
    useradd --uid $USER_ID --gid $GROUP_ID -m -s /bin/bash david && \
    echo "david ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Configurar locale
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# ROS 2 Humble Setup
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-rclcpp \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-rosidl-default-generators \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    && rm -rf /var/lib/apt/lists/*

#Instala dependencias adicionales de ROS 2
RUN apt-get install -y ros-humble-image-transport-plugins

RUN apt-get update && apt-get install -y \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-camera-calibration-parsers \
    && rm -rf /var/lib/apt/lists/*


# Instalar Livox-SDK2 desde el código fuente
WORKDIR /tmp
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
    cd Livox-SDK2 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/Livox-SDK2


# Define entrypoint para ROS 2
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && if [ -f install/setup.bash ]; then source install/setup.bash; fi && exec \"$@\"", "--"]

