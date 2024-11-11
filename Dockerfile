FROM ubuntu:24.04
USER root

# Install git
RUN apt-get update && apt-get install -y git

# Clone dependencies
RUN git clone -b llvmorg-18.1.4 --depth 1 https://github.com/llvm/llvm-project.git /llvm
RUN git clone -b cost-action https://github.com/saulshanabrook/egg-smol.git /egglog

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential clang cmake lld ninja-build \
    python3 python3-pip python3-venv

# Copy the DialEGG source code
COPY . /dialegg

# Add run permissions to scripts
RUN chmod +x /dialegg/build.sh
RUN chmod +x /dialegg/run.sh