import re
from pathlib import Path

import matplotlib.pyplot as plt


def extract_timing(filepath: Path):
    with filepath.open("r") as f:
        first_line = f.readline().strip()

    pattern = r"# workers=(\d+), N=(\d+), elapsed_seconds=([0-9.]+)"
    match = re.match(pattern, first_line)

    if match is None:
        raise ValueError(f"Could not parse timing line in file: {filepath}")

    workers = int(match.group(1))
    n_floorplans = int(match.group(2))
    elapsed = float(match.group(3))

    return workers, n_floorplans, elapsed


def load_results(folder: Path):
    files = sorted(folder.glob("n=50_processes=*.txt"))

    if not files:
        raise FileNotFoundError(f"No timing files found in {folder}")

    results = []
    for file in files:
        workers, n_floorplans, elapsed = extract_timing(file)
        results.append((workers, elapsed))

    results.sort(key=lambda x: x[0])
    return results


def compute_speedups(results):
    workers = [r[0] for r in results]
    times = [r[1] for r in results]

    baseline = times[0]
    speedups = [baseline / t for t in times]

    return workers, times, speedups


def print_table(label, workers, times, speedups):
    print(f"\n{label} results:")
    print("workers\telapsed_seconds\tspeedup")
    for w, t, s in zip(workers, times, speedups):
        print(f"{w}\t{t:.6f}\t{s:.4f}")


def main():
    static_folder = Path("../task5")
    dynamic_folder = Path(".")

    static_results = load_results(static_folder)
    dynamic_results = load_results(dynamic_folder)

    workers_static, times_static, speedups_static = compute_speedups(static_results)
    workers_dynamic, times_dynamic, speedups_dynamic = compute_speedups(dynamic_results)

    print_table("Static scheduling", workers_static, times_static, speedups_static)
    print_table("Dynamic scheduling", workers_dynamic, times_dynamic, speedups_dynamic)

    # Plot elapsed time comparison
    plt.figure(figsize=(8, 5))
    plt.plot(workers_static, times_static, marker="o", label="Static scheduling")
    plt.plot(workers_dynamic, times_dynamic, marker="o", label="Dynamic scheduling")
    plt.xlabel("Number of workers")
    plt.ylabel("Elapsed time (seconds)")
    plt.title("Elapsed time: static vs dynamic scheduling (N=50)")
    plt.grid(True)
    plt.xticks(workers_static)
    plt.legend()
    plt.tight_layout()
    plt.savefig("elapsed_static_vs_dynamic.png", dpi=200)
    plt.show()

    # Plot speed-up comparison
    plt.figure(figsize=(8, 5))
    plt.plot(workers_static, speedups_static, marker="o", label="Static scheduling")
    plt.plot(workers_dynamic, speedups_dynamic, marker="o", label="Dynamic scheduling")
    plt.plot(workers_static, workers_static, linestyle="--", label="Ideal speed-up")
    plt.xlabel("Number of workers")
    plt.ylabel("Speed-up")
    plt.title("Speed-up: static vs dynamic scheduling (N=50)")
    plt.grid(True)
    plt.xticks(workers_static)
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup_static_vs_dynamic.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()