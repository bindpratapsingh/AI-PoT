#!/usr/bin/env python3
"""
analyze_results.py

Usage:
  # old style: PoT vs AI-PoT
  python3 analyze_results.py logs/pot.csv logs/aipot.csv

  # new style: PoT vs RR vs AI-PoT
  python3 analyze_results.py logs/pot.csv logs/rr.csv logs/aipot.csv
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


# ---------------------------------------------
# Basic summary statistics for one CSV
# ---------------------------------------------
def summary(df):
    lat = df['actual_ms'].values  # actual task completion times (ms)
    out = {
        'n': int(len(lat)),
        'mean': float(np.mean(lat)),
        'p50': float(np.percentile(lat, 50)),
        'p90': float(np.percentile(lat, 90)),
        'p95': float(np.percentile(lat, 95)),
        'p99': float(np.percentile(lat, 99)),
    }
    return out


def pct_improvement(baseline, new):
    """
    Positive result means NEW is faster (lower latency) than baseline.
    Improvement (%) = (baseline - new) / baseline * 100
    """
    if baseline <= 0:
        return float('nan')
    return (baseline - new) / baseline * 100.0


# ---------------------------------------------
# Compare exactly two policies (old behavior)
# ---------------------------------------------
def compare_two(csv1, csv2, name1="PoT", name2="AI-PoT"):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    s1 = summary(df1)
    s2 = summary(df2)

    print(f"--- {name1} ---")
    for k, v in s1.items():
        print(f"{k:>5}: {v:.3f}")

    print(f"\n--- {name2} ---")
    for k, v in s2.items():
        print(f"{k:>5}: {v:.3f}")

    print(f"\n--- Delta ({name2} minus {name1}) ---")
    for k in s1:
        if k == 'n':
            continue
        delta = s2[k] - s1[k]
        print(f"{k:>5}: {delta:.3f}")

    # Also show percentage improvement for key metrics
    print(f"\n--- % Improvement ({name2} vs {name1}, positive = {name2} is faster) ---")
    for k in ['mean', 'p50', 'p90', 'p95', 'p99']:
        imp = pct_improvement(s1[k], s2[k])
        print(f"{k:>5}: {imp:.2f}%")

    # Minimal plotting for 2-way case (optional)
    os.makedirs("plots", exist_ok=True)

    metrics = ['mean', 'p95', 'p99']
    for m in metrics:
        plt.figure()
        vals = [s1[m], s2[m]]
        labels = [name1, name2]
        plt.bar(labels, vals)
        plt.ylabel("Latency (ms)")
        plt.title(f"{m} latency: {name1} vs {name2}")
        plt.tight_layout()
        outpath = os.path.join("plots", f"bar_{m}_{name1}_{name2}.png")
        plt.savefig(outpath)
        plt.close()


# ---------------------------------------------
# Compare three policies: PoT, RR, AI-PoT
# ---------------------------------------------
def compare_three(pot_csv, rr_csv, aipot_csv):
    pot   = pd.read_csv(pot_csv)
    rr    = pd.read_csv(rr_csv)
    aipot = pd.read_csv(aipot_csv)

    s_pot   = summary(pot)
    s_rr    = summary(rr)
    s_aipot = summary(aipot)

    # 1) Print raw stats
    print("=== Summary Statistics ===\n")

    def print_block(name, s):
        print(f"--- {name} ---")
        for k, v in s.items():
            print(f"{k:>5}: {v:.3f}")
        print()

    print_block("Classic PoT", s_pot)
    print_block("Round Robin", s_rr)
    print_block("AI-PoT",      s_aipot)

    # 2) Pairwise improvements
    print("=== Pairwise % Improvement (positive means row is faster than column) ===")
    # We'll build a small table for mean, p95, p99
    metrics = ['mean', 'p90', 'p95', 'p99']
    policies = {
        'PoT': s_pot,
        'RR': s_rr,
        'AI-PoT': s_aipot,
    }

    # For each metric, show pairwise improvements
    for m in metrics:
        print(f"\nMetric: {m}")
        print(f"{'':10s}PoT        RR         AI-PoT")
        for name_row, s_row in policies.items():
            line = f"{name_row:10s}"
            for name_col, s_col in policies.items():
                if name_row == name_col:
                    cell = "   -    "
                else:
                    imp = pct_improvement(s_col[m], s_row[m])
                    cell = f"{imp:7.2f}%"
                line += cell + " "
            print(line)

    # 3) Specific headline statements
    print("\n=== Headline Comparisons ===")
    def headline(basename, s_base, name2, s2):
        for m in ['mean', 'p90', 'p95', 'p99']:
            imp = pct_improvement(s_base[m], s2[m])
            print(f"{name2} vs {basename}: {m} improved by {imp:.2f}%")

    print()
    headline("PoT", s_pot, "AI-PoT", s_aipot)
    print()
    headline("RR",  s_rr,  "AI-PoT", s_aipot)
    print()
    headline("PoT", s_pot, "RR",     s_rr)

    # 4) Plots
    os.makedirs("plots", exist_ok=True)

    # 4.1 Bar chart: mean latency
    plt.figure()
    labels = ['PoT', 'RR', 'AI-PoT']
    means  = [s_pot['mean'], s_rr['mean'], s_aipot['mean']]
    plt.bar(labels, means)
    plt.ylabel("Mean Latency (ms)")
    plt.title("Mean latency comparison")
    plt.tight_layout()
    plt.savefig("plots/mean_latency_comparison.png")
    plt.close()

    # 4.2 Bar chart: P95 and P99
    plt.figure()
    x = np.arange(len(labels))
    width = 0.35
    p95_vals = [s_pot['p95'], s_rr['p95'], s_aipot['p95']]
    p99_vals = [s_pot['p99'], s_rr['p99'], s_aipot['p99']]

    plt.bar(x - width/2, p95_vals, width, label='P95')
    plt.bar(x + width/2, p99_vals, width, label='P99')
    plt.xticks(x, labels)
    plt.ylabel("Latency (ms)")
    plt.title("Tail latencies (P95 & P99)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/tail_latency_p95_p99.png")
    plt.close()

    # 4.3 CDF comparison
    def cdf_xy(lat):
        lat_sorted = np.sort(lat)
        n = len(lat_sorted)
        y = np.linspace(0, 1, n, endpoint=True)
        return lat_sorted, y

    pot_x, pot_y     = cdf_xy(pot['actual_ms'].values)
    rr_x, rr_y       = cdf_xy(rr['actual_ms'].values)
    aipot_x, aipot_y = cdf_xy(aipot['actual_ms'].values)

    plt.figure()
    plt.plot(pot_x, pot_y, label='PoT')
    plt.plot(rr_x, rr_y, label='RR')
    plt.plot(aipot_x, aipot_y, label='AI-PoT')
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title("CDF of latencies (PoT vs RR vs AI-PoT)")
    plt.xscale('log')  # log scale to highlight tails
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/cdf_pot_rr_aipot.png")
    plt.close()

    # 4.4 Optional: log-histogram of latencies for each policy
    plt.figure()
    plt.hist(np.log10(pot['actual_ms'] + 1e-6), bins=50, alpha=0.5, label='PoT')
    plt.hist(np.log10(rr['actual_ms'] + 1e-6),  bins=50, alpha=0.5, label='RR')
    plt.hist(np.log10(aipot['actual_ms'] + 1e-6), bins=50, alpha=0.5, label='AI-PoT')
    plt.xlabel("log10(latency_ms)")
    plt.ylabel("Count")
    plt.title("Log-scaled latency histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/hist_log_latency.png")
    plt.close()

    print("\nPlots written under ./plots/")
    print("  - mean_latency_comparison.png")
    print("  - tail_latency_p95_p99.png")
    print("  - cdf_pot_rr_aipot.png")
    print("  - hist_log_latency.png")


# ---------------------------------------------
# Entry point
# ---------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 3:
        # two-CSV mode (backward compatible)
        compare_two(sys.argv[1], sys.argv[2], name1="PoT", name2="AI-PoT")
    elif len(sys.argv) == 4:
        # three-CSV mode: PoT, RR, AI-PoT
        pot_csv   = sys.argv[1]
        rr_csv    = sys.argv[2]
        aipot_csv = sys.argv[3]
        compare_three(pot_csv, rr_csv, aipot_csv)
    else:
        # default: same as old behavior (pot vs aipot)
        compare_two('logs/pot.csv', 'logs/aipot.csv', name1="PoT", name2="AI-PoT")
