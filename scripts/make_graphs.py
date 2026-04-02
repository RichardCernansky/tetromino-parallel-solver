#!/usr/bin/env python3
"""
graphs.py — generate speedup and efficiency graphs from results.csv
Usage: python3 graphs.py results.csv
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_file = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
df = pd.read_csv(csv_file)
df["cores"] = df["cores"].astype(int)
df["wall_time"] = df["wall_time"].astype(float)

colors = {"data": "#2196F3", "task": "#FF9800", "mpi": "#4CAF50"}
markers = {"data": "o", "task": "s", "mpi": "D"}

for map_name in sorted(df["map"].unique()):
    m = df[df["map"] == map_name]

    # Get sequential baseline
    seq = m[m["variant"] == "seq"]
    if len(seq) == 0:
        print(f"WARNING: no sequential baseline for {map_name}, skipping")
        continue
    seq_time = seq["wall_time"].values[0]

    max_cores = m["cores"].max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Map: {map_name}  (sequential: {seq_time:.1f}s)", fontsize=14, fontweight="bold")

    # ---- Plot 1: Wall time ----
    ax = axes[0]
    for variant in ["data", "task", "mpi"]:
        v = m[m["variant"] == variant].sort_values("cores")
        if len(v) == 0:
            continue
        ax.plot(v["cores"], v["wall_time"],
                marker=markers[variant], color=colors[variant],
                label=variant, linewidth=2, markersize=6)

    ax.axhline(y=seq_time, color="gray", linestyle="--", alpha=0.5, label=f"sequential ({seq_time:.1f}s)")
    ax.set_xlabel("Cores", fontsize=11)
    ax.set_ylabel("Wall time [s]", fontsize=11)
    ax.set_title("Execution Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    # ---- Plot 2: Speedup ----
    ax = axes[1]
    ax.plot([1, max_cores], [1, max_cores], "k--", alpha=0.3, label="Ideal linear")

    for variant in ["data", "task", "mpi"]:
        v = m[m["variant"] == variant].sort_values("cores")
        if len(v) == 0:
            continue
        speedup = seq_time / v["wall_time"]
        ax.plot(v["cores"], speedup,
                marker=markers[variant], color=colors[variant],
                label=variant, linewidth=2, markersize=6)

    ax.set_xlabel("Cores", fontsize=11)
    ax.set_ylabel("Speedup S(p)", fontsize=11)
    ax.set_title("Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Efficiency ----
    ax = axes[2]
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="Ideal")

    for variant in ["data", "task", "mpi"]:
        v = m[m["variant"] == variant].sort_values("cores")
        if len(v) == 0:
            continue
        speedup = seq_time / v["wall_time"]
        efficiency = speedup / v["cores"]
        ax.plot(v["cores"], efficiency,
                marker=markers[variant], color=colors[variant],
                label=variant, linewidth=2, markersize=6)

    ax.set_xlabel("Cores", fontsize=11)
    ax.set_ylabel("Efficiency E(p)", fontsize=11)
    ax.set_title("Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    out = f"graph_{map_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

print("Done")