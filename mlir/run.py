import os
import subprocess
import sys

## Usualy the process is for interpreting
# mlir-opt <FLAGS> <MLIR_FILE_PATH/MLIR_FILE_NAME.mlir> -o <MLIR_FILE_PATH/MLIR_FILE_NAME.ll.mlir>
# mlir-translate --mlir-to-llvmir <MLIR_FILE_PATH/MLIR_FILE_NAME.ll.mlir> -o <MLIR_FILE_PATH/MLIR_FILE_NAME.ll>
# lli --dlopen=/Users/aziz/dev/lib/llvm/build/lib/libmlir_c_runner_utils.dylib <MLIR_FILE_PATH/MLIR_FILE_NAME.ll>

## Process for compiling and running
# mlir-opt <FLAGS> <MLIR_FILE_PATH/MLIR_FILE_NAME.mlir> -o <MLIR_FILE_PATH/MLIR_FILE_NAME.ll.mlir>
# mlir-translate --mlir-to-llvmir <MLIR_FILE_PATH/MLIR_FILE_NAME.ll.mlir> -o <MLIR_FILE_PATH/MLIR_FILE_NAME.ll>
# llc -filetype=obj <MLIR_FILE_PATH/MLIR_FILE_NAME.ll> -o <MLIR_FILE_PATH/MLIR_FILE_NAME.o>
# clang <MLIR_FILE_PATH/MLIR_FILE_NAME.o> -o <MLIR_FILE_PATH/MLIR_FILE_NAME> -L/Users/aziz/dev/lib/llvm/build/lib -lmlir_c_runner_utils -Wl,-rpath,/Users/aziz/dev/lib/llvm/build/lib
# ./<MLIR_FILE_PATH/MLIR_FILE_NAME>

def run_mlir_file(mlir_file, opt="-O3", extra_passes=None, verbose=True):
    mlir_filepath = os.path.dirname(mlir_file)
    mlir_filename = os.path.basename(mlir_file).replace(".mlir", "") # filename wihtout path and extension
    run_dir = os.path.join(mlir_filepath, "run")

    # Create run directory if it doesn't exist
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    flags = [
        "--convert-elementwise-to-linalg",
        "--convert-tensor-to-linalg",
        "--convert-linalg-to-loops",
        "--one-shot-bufferize=bufferize-function-boundaries=true",
        "--convert-linalg-to-loops",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-index-to-llvm",
        "--convert-math-to-llvm",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-func-to-llvm",
        "--finalizing-bufferize",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    if extra_passes:
        flags += extra_passes
    
    flags_str = " ".join(flags)

    # Lower the MLIR code to LLVM IR
    mlir_ll_file = os.path.join(run_dir, f"{mlir_filename}.ll.mlir")
    if verbose:
        print(f"Creating {mlir_ll_file}")
    # subprocess.run(["mlir-opt", *flags, mlir_file, "-o", mlir_ll_file])
    subprocess.run(f"mlir-opt {flags_str} {mlir_file} -o {mlir_ll_file}", shell=True, check=True)  # mlir-opt $FLAGS $MLIR_FILE -o $mlir_ll_file

    # Translate the MLIR code to LLVM IR
    ll_file = os.path.join(run_dir, f"{mlir_filename}.ll")
    if verbose:
        print(f"Creating {ll_file}")
    subprocess.run(["mlir-translate", "--mlir-to-llvmir", mlir_ll_file, "-o", ll_file]) # mlir-translate --mlir-to-llvmir $mlir_ll_file -o $ll_file

    # opt the LLVM IR code
    opt_ll_file = os.path.join(run_dir, f"{mlir_filename}.opt.ll")
    if verbose:
        print(f"Creating {opt_ll_file}")
    subprocess.run(["opt", opt, ll_file, "-S", "-o", opt_ll_file]) # opt -O3 $ll_file -S -o $opt_ll_file

    # Create executable from the LLVM IR code
    o_file = os.path.join(run_dir, f"{mlir_filename}.o")
    if verbose:
        print(f"Creating {o_file}")
    subprocess.run(["llc", "-filetype=obj", opt, opt_ll_file, "-o", o_file]) # llc -filetype=obj -O3 $opt_ll_file -o $o_file

    exec_file = os.path.join(run_dir, mlir_filename)
    if verbose:
        print(f"Creating {exec_file}")

    lib_path = "/Users/aziz/dev/lib/llvm/build/lib"
    lib = "mlir_c_runner_utils"

    # clang -O3 $o_file -o $exe_file -L/Users/aziz/dev/lib/llvm/build/lib -lmlir_c_runner_utils -Wl,-rpath,/Users/aziz/dev/lib/llvm/build/lib
    subprocess.run(["clang", opt, o_file, "-o", exec_file, 
                    f"-L{lib_path}", f"-l{lib}", "-L/Users/aziz/dev/current/dialegg/mlir/run", "-lutil",
                    f"-Wl,-rpath,{lib_path}", "-Wl,-rpath,/Users/aziz/dev/current/dialegg/mlir/run"])

    # Run the executable
    result = subprocess.run(exec_file, shell=True, text=True, capture_output=True) # exe_file
    if verbose:
        print(f"Running the executable {exec_file}: return code {result.returncode}")
    if result.returncode != 0:
        sys.exit(result.returncode)  # exit if there was an error
    
    out = result.stdout
    if verbose:  # Save executable output
        out_file = os.path.join(mlir_filepath, f"{mlir_filename}.out")
        with open(out_file, "w") as f:
            f.write(out)

    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <mlir_filepath/mlir_filename.mlir>")
        sys.exit(1)
    
    mlir_file = sys.argv[1]
    out = run_mlir_file(mlir_file, verbose=True)  # format "53926 us -> 0.053926 s"
    # print(out)