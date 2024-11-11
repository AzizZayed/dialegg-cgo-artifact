# DialEgg: MLIR + Equality Saturation
This tool proposes a dialect-agnostic way to apply equality saturation optimizations to LLVM's Multi-Level IR ([MLIR](https://mlir.llvm.org/)).
The equality saturation engine used is [Egglog](https://github.com/egraphs-good/egglog).

This artifact ...

## Steps to build

### Hardware Requirements

### Run Docker Container
You need docker on your system. Follow the instructions [here](https://docs.docker.com/get-started/get-docker/).

Build the docker image and give it a name:
```bash
docker build -t dialegg-image .
```

Then run and enter the container
```bash
docker run --name dialegg-ubuntu --rm -i -t dialegg-image bash
```

Within the container, you run the build script that build LLVM with MLIR, Egglog, and DialEgg. This will take a long time.
```bash
cd dialegg
./build.sh
```

Each benchmarks is a subdirectory of the `bench/` directory.
Each subdirectory has the main MLIR file and an egg file.

When you run the optimizer, you will find each subdirectory is populated with the optimized versions using the different methods mentioned in section 8 of the paper and figure 3.
```bash
source venv/bin/activate
python bench/opt.py
```

When you run the follwing commands, the data and plot for figure 3 is generated
```bash
python bench/bench.py
python bench/plot.py
```

To get the compilation time data in table 2, look at the output of the following command:
```bash
python test/timer.py
```

## Reusability
You can use this for any MLIR file, any dialect.