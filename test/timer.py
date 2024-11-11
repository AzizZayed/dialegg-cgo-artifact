import subprocess
import re
import os
import statistics
import pandas as pd

def eq_sat_time(file_path, interest_function, data, nruns=10):
    # check if file exists
    if not os.path.exists(file_path):
        print(f"Error: file {file_path} does not exist")
        return

    if file_path not in data:
        data[file_path] = {}

    times = {}
    saturation_times = []
    for i in range(nruns):
        extra = ["-egg", "test/nmm/nmm.egg"] if "nmm" in file_path else []
        res = subprocess.run(["./build/egg-opt", "--mlir-disable-threading", "--eq-sat", file_path] + extra, capture_output=True, text=True)

        # find the text "Done running on function: " + interest_function, then print it and the 6 lines after it
        lines = res.stdout.split("\n")
        for i, line in enumerate(lines):
            if line == f"Done running on function: {interest_function}":
                for j, line in enumerate(lines[i:i+7]):
                    for keyword in ['mlirToEgglogTime', 'egglogExecTime', 'egglogToMlirTime']:
                        if keyword in line:
                            # line is like : "mlirToEgglogCpuTime = 2.467200e-02s" so get the number
                            total_time = float(line.split("=")[1][:-1].strip())
                            
                            if keyword not in times:
                                times[keyword] = []
                            
                            times[keyword].append(total_time)
                            break
        
        # find "Running on function: " + interest_function, then after that, find the line "Ruleset rules: search ..."
        for i, line in enumerate(lines):
            if line == f"Running on function: {interest_function}":
                for j, line in enumerate(lines[i:]):
                    if "Ruleset rules: search" in line:
                        # this line is like: "Ruleset rules: search 0.002s, apply 0.001s, rebuild 0.001s" so sum the numbers to get the total time
                        # print(line)
                        total_time = sum([float(x) for x in re.findall(r"\d+\.\d+", line)])
                        saturation_times.append(total_time)
                        break
    
    data[file_path]["egglogSaturateTime"] = statistics.median(saturation_times)
    for key in times:
        data[file_path][key] = statistics.median(times[key])


def canon_time(file_path, data, nruns=10):
    if file_path not in data:
        data[file_path] = {}

    times = []
    for i in range(nruns):
        res = subprocess.run(["mlir-opt", "--canonicalize", "-mlir-timing", file_path], capture_output=True, text=True)

        lines = res.stderr.split("\n")
        for i, line in enumerate(lines):
            if "Canonicalizer" in line:
                # line is like "0.0233 ( 68.6%)    Canonicalizer" so get the first number
                total_time = float(re.findall(r"\d+\.\d+", line)[0])
                times.append(total_time)
    
    data[file_path]["Canon"] = statistics.median(times)
        

def cpp_time(file_path, data, nruns=10):
    if file_path not in data:
        data[file_path] = {}

    times = []
    for i in range(nruns):
        res = subprocess.run(["./build/egg-opt", "--mlir-disable-threading", "--matmul-associate", "-mlir-timing", file_path], capture_output=True, text=True)

        lines = res.stderr.split("\n")
        for i, line in enumerate(lines):
            if "MatrixMultiplyAssociatePass" in line:
                # line is like "0.0233 ( 68.6%)    MatrixMultiplyAssociatePass" so get the first number
                total_time = float(re.findall(r"\d+\.\d+", line)[0])
                times.append(total_time)
    
    data[file_path]["C++ impl"] = statistics.median(times)


def main():
    data = {}

    eq_sat_time("test/image_conversion/image_conversion.mlir", "rgb_to_grayscale", data)
    eq_sat_time("test/vector_norm/vector_norm.mlir", "normalize_vector", data)
    eq_sat_time("test/polynomial/polynomial.mlir", "poly_eval_3", data)

    canon_time("test/image_conversion/image_conversion.mlir", data)
    canon_time("test/vector_norm/vector_norm.mlir", data)
    canon_time("test/polynomial/polynomial.mlir", data)
    # canon_time("test/3mm/3mm.mlir", data)

    for n in [2, 3, 10, 20, 40, 80]:
        file  = f"test/nmm/{n}mm.mlir"
        eq_sat_time(file, f"_{n}mm", data)
        canon_time(file, data)
        cpp_time(file, data)

    print(data)
    df = pd.DataFrame(data).T
    df = df * 1000

    df.rename(index={
        "test/image_conversion/image_conversion.mlir": "Image Conversion",
        "test/vector_norm/vector_norm.mlir": "Vector Norm",
        "test/polynomial/polynomial.mlir": "Polynomial",
        "test/nmm/2mm.mlir": "2mm",
        "test/nmm/3mm.mlir": "3mm",
        "test/nmm/10mm.mlir": "10mm",
        "test/nmm/20mm.mlir": "20mm",
        "test/nmm/40mm.mlir": "40mm",
        "test/nmm/80mm.mlir": "80mm",
    }, inplace=True)

    df.rename(columns={
        "egglogSaturateTime": "Saturate Time (ms)",
        "mlirToEgglogTime": "MLIR to Egglog Time (ms)",
        "egglogExecTime": "Egglog Execution Time (ms)",
        "egglogToMlirTime": "Egglog to MLIR Time (ms)",
        "Canon": "Canonicalize Time (ms)",
        "C++ impl": "C++ Implementation Time (ms)",
    }, inplace=True)

    print()
    print(df)
    df.to_csv("test/times.csv")


if __name__ == "__main__":
    main()