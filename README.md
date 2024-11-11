# DialEgg: MLIR + Equality Saturation
This tool proposes a dialect-agnostic way to apply equality saturation optimizations to LLVM's Multi-Level IR ([MLIR](https://mlir.llvm.org/)).
The equality saturation engine used is [Egglog](https://github.com/egraphs-good/egglog).

This artifact ...

## Steps to build

### Hardware Requirements

### Run Docker Container
You need docker on your system.

Build the docker image and give it a name:
```bash
docker build -t dialegg-image .
```

Then run and enter the container
```bash
docker run --name dialegg-ubuntu --rm -i -t dialegg-image bash
```

## Reusability
You can use this for any MLIR file, any dialect.