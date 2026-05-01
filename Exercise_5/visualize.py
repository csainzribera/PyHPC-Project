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


def main():
    files = sorted(Path(".").glob("n=50_processes=*.txt"))

    if not files:
        raise FileNotFoundError("No files matching 'n=50_processes=*.txt' were found.")

    results = []
    for file in files:
        workers, n_floorplans, elapsed = extract_timing(file)
        results.append((workers, elapsed))

    results.sort(key=lambda x: x[0])

    workers = [r[0] for r in results]
    elapsed_times = [r[1] for r in results]

    # Baseline = same parallel script with 1 worker
    t1 = elapsed_times[0]
    speedups = [t1 / t for t in elapsed_times]

    # Plot 1: elapsed time
    plt.figure(figsize=(8, 5))
    plt.plot(workers, elapsed_times, marker="o")
    plt.xlabel("Number of workers")
    plt.ylabel("Elapsed time (seconds)")
    plt.title("Elapsed time vs number of workers (N=50)")
    plt.grid(True)
    plt.xticks(workers)
    plt.tight_layout()
    plt.savefig("elapsed_vs_workers.png", dpi=200)
    plt.show()

    # Plot 2: speed-up
    plt.figure(figsize=(8, 5))
    plt.plot(workers, speedups, marker="o", label="Measured speed-up")
    plt.plot(workers, workers, linestyle="--", label="Ideal speed-up")
    plt.xlabel("Number of workers")
    plt.ylabel("Speed-up")
    plt.title("Speed-up vs number of workers (N=50)")
    plt.grid(True)
    plt.xticks(workers)
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup_vs_workers.png", dpi=200)
    plt.show()

    # Print a small table
    print("\nResults:")
    print("workers\telapsed_seconds\tspeedup")
    for w, t, s in zip(workers, elapsed_times, speedups):
        print(f"{w}\t{t:.6f}\t{s:.4f}")


if __name__ == "__main__":
    main()