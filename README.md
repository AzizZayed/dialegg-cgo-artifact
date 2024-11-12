# DialEgg: MLIR + Equality Saturation
This tool proposes a dialect-agnostic way to apply equality saturation optimizations to LLVM's Multi-Level IR ([MLIR](https://mlir.llvm.org/)).
The equality saturation engine used is [Egglog](https://github.com/egraphs-good/egglog).

This artifact contains a source tree and a Dockerfile.
With docker, we will build a container that includes DialEgg and its dependencies, LLVM, MLIR and Egglog.
This container is designed to reproduce this paper’s experimental results.

## Steps to build

### Software Requirements
- A Docker version 27.3.1 installation. Follow the instructions [here](https://docs.docker.com/get-started/get-docker/).

### Hardware Requirements
- Preferably an ARM SoC that is powerful enough to build LLVM and MLIR from source (i.e. preferably 8 or more cores).
- At least 16GB of ram
- At least 15GB of free disk space

### Run Docker Container
Build the docker image and give it a name:
```bash
docker build -t dialegg-image .
```

Then run and enter the container
```bash
docker run --name dialegg-ubuntu --rm -i -t dialegg-image bash
```

Within the container, you run the build script that build LLVM with MLIR, Egglog, and DialEgg. LLVM is known to take a long time to build so this build script will take a long time.
```bash
cd dialegg

./build.sh

source venv/bin/activate
source .env
```

Each benchmarks is a subdirectory of the `bench/` directory.
Each subdirectory has the main MLIR file and an egg file.

When you run the optimizer, you will find that each benchmark directory is populated with the optimized versions using the different methods mentioned in section 8 of the paper and figure 3.
```bash
python bench/opt.py
```

For example, for `bench/polynomial/polynomial.mlir`, there will be:
- `bench/polynomial/polynomial.eqsat.mlir` for equality saturation optimization
- `bench/polynomial/polynomial.canon.mlir` for canonicalization
- `bench/polynomial/polynomial.eqsat+canon.mlir` for both equality saturation optimization and canonicalization

When you run the following commands, the data is the output and figure 3 from the paper is generated in `bench/speedup.pdf`.
```bash
python bench/bench.py
python bench/plot.py
```

To get the compilation time data in table 2 of the paper, look at the output of the following command:
```bash
python test/timer.py
```

## Reusability
DialEgg can be used to optimize any MLIR file with equality saturation, as long as each operation has a single result.
The core of the artifact is `egg-opt`, the binary produced under the `build` directory.
This is an `mlir-opt` tool that can take any supported MLIR code, and a corresponding egg file and perform equality saturation-based optimization.