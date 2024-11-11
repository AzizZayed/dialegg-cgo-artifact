#!/bin/bash

# Build LLVM with MLIR
cd /llvm
echo "Building LLVM with MLIR"
mkdir build-release
cmake -S llvm -B build-release -G Ninja -DLLVM_ENABLE_PROJECTS="mlir" -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_LINKER=lld
cmake --build build-release
export LLVM="/llvm/build-release/bin"
cd ..

# Build Egglog
cd /egglog
echo "Building Egglog"
git checkout b6e1c96ed7335366e90056ea0a24ef425dfbb8fb
cargo build --release
export EGGLOG="/egglog/target/release"
cd ..

# Expose execs to PATH
export PATH=$LLVM:$EGGLOG:$PATH

# Build DialEgg
echo "Building DialEgg"
mkdir build
cmake -S . -B build
cmake --build build

# Install Python dependencies
python3 -m venv /dialegg/venv
source /dialegg/venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r /dialegg/requirements.txt