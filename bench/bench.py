import os
import re
import sys
sys.path.append('.')

from mlir.run import run_mlir_file
import pandas as pd
import statistics

def same_name_files(name, directory):
    return [f for f in os.listdir(directory) if f.startswith(name + ".") and f.endswith(".mlir")]

def benchmark_file(filename, directory):
    all_mlir_files = same_name_files(filename, directory)
    all_opt_levels = ["-O3"]
    n_runs = 10  # number of runs
    regex_time = re.compile(r"(\d+) us ->")  # format "53926 us -> 0.053926 s"

    # Persist data, cols are: the name of the opt type, and each opt level
    data = {
        "name": [],
        "opt": [],
    }
    for opt_level in all_opt_levels:
        data[opt_level] = []

    print(f"--------------------------- Running {filename} ---------------------------")
    print(f"Found {len(all_mlir_files)} files: {all_mlir_files}")

    for mlir_file in all_mlir_files:
        ext = mlir_file.split(".")[-2]
        if ext == filename:
            ext = "noopt"
        
        data["opt"].append(ext)
        data["name"].append(filename)
        for opt_level in all_opt_levels:

            times = []
            for i in range(n_runs):
                time = run_mlir_file(os.path.join(directory, mlir_file), opt=opt_level, verbose=False)
                match = regex_time.search(time)
                if match:
                    times.append(float(match.group(1)))
                else:
                    print(f"Error: no match")

            avg_time = statistics.mean(times)
            median_time = statistics.median(times)

            data[opt_level].append(median_time)
            print(f"[{ext}:{opt_level}] times:{times} us, avg:{avg_time} us, median:{median_time} us")

        print()

    print(data)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(directory, f"{filename}.csv"), index=False)

if __name__ == "__main__":
    benchmark_file("image_conversion", "bench/image_conversion")
    benchmark_file("vector_norm", "bench/vector_norm")
    benchmark_file("polynomial", "bench/polynomial")
    benchmark_file("2mm", "bench/2mm")
    benchmark_file("3mm", "bench/3mm")