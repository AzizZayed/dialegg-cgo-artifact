FROM arm64v8/ubuntu:24.04
USER root

# Install git
RUN apt-get update && apt-get install -y git

# Clone dependencies
RUN git clone -b llvmorg-18.1.4 --depth 1 https://github.com/llvm/llvm-project.git /llvm
RUN git clone -b cost-action https://github.com/saulshanabrook/egg-smol.git /egglog

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    vim \
    build-essential clang automake cmake make ccache lld ninja-build zlib1g-dev \
    python3 python3-pip python3-venv \
    cargo \
    texlive texlive-fonts-recommended texlive-fonts-extra fonts-linuxlibertine dvipng

# Copy the DialEGG source code
COPY . /dialegg

# Add run permissions to scripts
RUN chmod +x /dialegg/build.sh

# Set C/C++ compiler to clang
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++