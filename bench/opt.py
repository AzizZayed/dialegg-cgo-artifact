import subprocess

# generate optimized mlir files
def opt_file(file_path):
    file_path_no_ext = file_path.removesuffix(".mlir")

    # Create .canon.mlir file
    subprocess.run(["mlir-opt", "--mlir-disable-threading", "--canonicalize", file_path, "-o", f"{file_path_no_ext}.canon.mlir"])
    # Create .eqsat.mlir file
    subprocess.run(["./build/egg-opt", "--mlir-disable-threading", "--eq-sat", file_path, "-o", f"{file_path_no_ext}.eqsat.mlir"])
    # Create .canon+eqsat.mlir file
    subprocess.run(["./build/egg-opt", "--mlir-disable-threading", "--canonicalize", "--eq-sat", file_path, "-o", f"{file_path_no_ext}.canon+eqsat.mlir"])
    # Create .eqsat+canon.mlir file
    subprocess.run(["./build/egg-opt", "--mlir-disable-threading", "--eq-sat", "--canonicalize", file_path, "-o", f"{file_path_no_ext}.eqsat+canon.mlir"])

    if "mm" in file_path:
        # create extra .cpp.mlir files
        subprocess.run(["./build/egg-opt", "--mlir-disable-threading", "--matmul-associate", file_path, "-o", f"{file_path_no_ext}.cpp.mlir"])

def main():
    opt_file("bench/image_conversion/image_conversion.mlir")
    opt_file("bench/vector_norm/vector_norm.mlir")
    opt_file("bench/polynomial/polynomial.mlir")
    opt_file("bench/2mm/2mm.mlir")
    opt_file("bench/3mm/3mm.mlir")

if __name__ == "__main__":
    main()