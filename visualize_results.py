#!/usr/bin/env python3
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(pot_path, aipot_path):
    pot = pd.read_csv(pot_path)
    aipot = pd.read_csv(aipot_path)
    return pot, aipot


def basic_stats(df):
    lat = df["actual_ms"].values
    return {
        "n": len(lat),
        "mean": lat.mean(),
        "p50": np.quantile(lat, 0.50),
        "p90": np.quantile(lat, 0.90),
        "p95": np.quantile(lat, 0.95),
        "p99": np.quantile(lat, 0.99),
    }


def print_stats(pot_stats, aipot_stats):
    print("\n=== Summary stats (from visualize_results.py) ===")
    print("--- Classic PoT ---")
    for k, v in pot_stats.items():
        print(f"{k:>4}: {v:.3f}")
    print("\n--- AI-PoT ---")
    for k, v in aipot_stats.items():
        print(f"{k:>4}: {v:.3f}")

    print("\n--- Improvement (Reduction in Latency) ---")
    print(f"{'Metric':>6} | {'Delta (ms)':>10} | {'% Speedup':>13}")
    print("-" * 35)
    for k in pot_stats.keys():
        if k == "n": continue
        
        baseline = pot_stats[k]
        new_val = aipot_stats[k]
        diff = baseline - new_val # Positive diff means latency reduced
        
        # Avoid division by zero
        if baseline != 0:
            pct = (diff / baseline) * 100
        else:
            pct = 0.0
            
        print(f"{k:>6} | {diff:10.3f} | {pct:12.2f}%")


def ensure_figdir():
    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)
    return outdir


def get_focus_limit(pot_lat, aipot_lat, percentile=98):
    """
    Finds a reasonable upper limit for the plots by looking at the 
    specified percentile of the combined data. This cuts off extreme outliers.
    """
    all_lat = np.concatenate([pot_lat, aipot_lat])
    return np.percentile(all_lat, percentile)


def plot_histograms(pot, aipot, outdir):
    pot_lat = pot["actual_ms"].values
    aipot_lat = aipot["actual_ms"].values

    # FIX: Focus on the main body of data (0 to p98) instead of the max outlier
    limit = get_focus_limit(pot_lat, aipot_lat, percentile=98)
    bins = np.linspace(0, limit, 60)

    plt.figure(figsize=(8, 5))
    plt.hist(pot_lat, bins=bins, density=True, alpha=0.5, label="PoT")
    plt.hist(aipot_lat, bins=bins, density=True, alpha=0.5, label="AI-PoT")

    plt.xlabel("Latency (ms)")
    plt.ylabel("Density")
    plt.title("Latency distribution (Zoomed to 98th percentile)")
    plt.xlim(0, limit)  # Force the view to cut off the long tail
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = os.path.join(outdir, "latency_histogram.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def plot_histograms_logx(pot, aipot, outdir):
    pot_lat = pot["actual_ms"].values
    aipot_lat = aipot["actual_ms"].values

    # Avoid zeros for log-scale
    pot_lat = pot_lat[pot_lat > 0]
    aipot_lat = aipot_lat[aipot_lat > 0]

    all_lat = np.concatenate([pot_lat, aipot_lat])
    
    # FIX: Even in log scale, super distant outliers can compress the view.
    # We clip the view to the 99.5th percentile.
    upper_limit = np.percentile(all_lat, 99.5)
    
    bins = np.logspace(
        np.log10(all_lat.min()),
        np.log10(upper_limit),
        60,
    )

    plt.figure(figsize=(8, 5))
    plt.hist(pot_lat, bins=bins, density=True, alpha=0.5, label="PoT")
    plt.hist(aipot_lat, bins=bins, density=True, alpha=0.5, label="AI-PoT")

    plt.xscale("log")
    plt.xlabel("Latency (ms, log-scale)")
    plt.ylabel("Density")
    plt.title("Latency distribution (log-x)")
    plt.xlim(all_lat.min(), upper_limit) # Ensure outliers don't stretch axis
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    path = os.path.join(outdir, "latency_histogram_logx.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def plot_cdf(pot, aipot, outdir):
    pot_lat = np.sort(pot["actual_ms"].values)
    aipot_lat = np.sort(aipot["actual_ms"].values)

    pot_y = np.linspace(0, 1, len(pot_lat), endpoint=False)
    aipot_y = np.linspace(0, 1, len(aipot_lat), endpoint=False)

    # FIX: Clip the x-axis to zoom in on the "curve" part
    limit = get_focus_limit(pot_lat, aipot_lat, percentile=98)

    plt.figure(figsize=(8, 5))
    plt.plot(pot_lat, pot_y, label="PoT")
    plt.plot(aipot_lat, aipot_y, label="AI-PoT")

    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title("CDF of latencies (Zoomed)")
    plt.xlim(0, limit)  # Crucial for visibility
    plt.grid(True, alpha=0.3)
    plt.legend()

    path = os.path.join(outdir, "latency_cdf.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def plot_bar_quantiles(pot_stats, aipot_stats, outdir):
    labels = ["mean", "p50", "p90", "p95", "p99"]
    pot_vals = [pot_stats[k] for k in labels]
    aipot_vals = [aipot_stats[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, pot_vals, width, label="PoT")
    plt.bar(x + width / 2, aipot_vals, width, label="AI-PoT")

    plt.xticks(x, labels)
    plt.ylabel("Latency (ms)")
    plt.title("Summary metrics: PoT vs AI-PoT")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    path = os.path.join(outdir, "summary_quantiles_bar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def plot_size_vs_latency(pot, aipot, outdir, max_points=5000):
    pot2 = pot.copy()
    aipot2 = aipot.copy()
    pot2["mode"] = "PoT"
    aipot2["mode"] = "AI-PoT"

    df = pd.concat([pot2, aipot2], ignore_index=True)

    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)

    # FIX: Calculate Zoom Limits
    lat_limit = np.percentile(df["actual_ms"], 98)
    size_limit = np.percentile(df["size"], 99)

    plt.figure(figsize=(8, 5))
    for mode, sub in df.groupby("mode"):
        plt.scatter(
            sub["size"],
            sub["actual_ms"],
            s=15, # Increased dot size slightly for visibility
            alpha=0.5,
            label=mode,
        )

    plt.xlabel("Task size (bytes)")
    plt.ylabel("Latency (ms)")
    plt.title("Task size vs latency (Zoomed)")
    
    # Apply zoom
    plt.ylim(0, lat_limit)
    plt.xlim(0, size_limit)
    
    plt.grid(True, alpha=0.3)
    plt.legend()

    path = os.path.join(outdir, "size_vs_latency_scatter.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved {path}")


def main():
    if len(sys.argv) == 3:
        pot_path = sys.argv[1]
        aipot_path = sys.argv[2]
    else:
        # Defaults
        pot_path = "logs/pot.csv"
        aipot_path = "logs/aipot.csv"
        print(f"No CSVs passed; defaulting to {pot_path} and {aipot_path}")

    if not (os.path.exists(pot_path) and os.path.exists(aipot_path)):
        print(f"ERROR: Could not find {pot_path} or {aipot_path}")
        sys.exit(1)

    pot, aipot = load_data(pot_path, aipot_path)

    pot_stats = basic_stats(pot)
    aipot_stats = basic_stats(aipot)
    print_stats(pot_stats, aipot_stats)

    outdir = ensure_figdir()
    plot_histograms(pot, aipot, outdir)
    plot_histograms_logx(pot, aipot, outdir)
    plot_cdf(pot, aipot, outdir)
    plot_bar_quantiles(pot_stats, aipot_stats, outdir)
    plot_size_vs_latency(pot, aipot, outdir)


if __name__ == "__main__":
    main()