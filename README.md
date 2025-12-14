# AI-Assisted Task Scheduling using Power-of-Two Choices (AI-PoT)

This project implements and evaluates an **AI-enhanced Power-of-Two-Choices (PoT) scheduling algorithm** for distributed systems.  
The goal is to reduce **tail latency (P95/P99)** under **heterogeneous nodes** and **heavy-tailed workloads**, where classical queue-length–based schedulers fail.

The system is implemented using **MPI (OpenMPI)** to simulate a distributed cluster and integrates a **machine learning–based task execution time predictor** into the scheduling decision.

---

## Motivation

Classic Power-of-Two-Choices (PoT) assigns tasks based on **queue length**, assuming all tasks are roughly equal in cost.  
In real distributed systems, this assumption breaks due to:

- Straggler nodes (slow or throttled machines)
- Heavy-tailed workloads (few large “elephant” tasks)
- Latency-sensitive applications where **P99 latency matters more than average latency**

**AI-PoT** augments PoT by scheduling based on **predicted remaining execution time**, not just task count.

---

## Key Contributions

- **AI-PoT Scheduler**  
  Extends PoT by incorporating ML-based task cost prediction.

- **Tail-Aware Machine Learning Model**  
  Uses linear regression (P90-focused Gradient Boosting distilled into a fast linear model) for tail-latency awareness with negligible runtime overhead.

- **Distributed Simulation with MPI**  
  Simulates a master–worker system with node heterogeneity, stragglers, and elephant–mice task distributions.

- **Comprehensive Evaluation**  
  Measures Mean, P50, P90, P95, and P99 latencies and compares AI-PoT against classic PoT.

---

## System Architecture

- **Master (Dispatcher)**
  - Generates tasks
  - Samples two workers (Power-of-Two)
  - Assigns tasks based on:
    - Queue length (PoT)
    - Predicted remaining work (AI-PoT)

- **Workers**
  - Execute CPU or I/O tasks
  - Simulate heterogeneity via slowdown factors
  - Report actual execution time

- **ML Model**
  - Trained offline using execution logs
  - Inference at runtime is O(1)

---

## Project Structure
├── scheduler.c # MPI-based scheduler (PoT + AI-PoT)

├── train_model.py # Tail-aware ML training and distillation

├── analyze_results.py # Latency percentile analysis

├── logs/ # Execution logs (CSV)

├── model/

│ └── coeffs.txt # Trained model coefficients

└── README.md


---

## How to Run

```bash

# 1. Compile Scheduler
mpicc -o scheduler scheduler.c -lm

# 2. Run Classic PoT
rm -f logs/*.csv
mpirun --oversubscribe -np 10 ./scheduler --mode pot --tasks 50000 --seed 42
cp logs/run_tasks.csv logs/pot.csv

# 3. Train AI Model
python3 train_model.py logs/pot.csv

This generates --> model/coeffs.txt

# 4. Run AI-PoT
rm -f logs/run_tasks.csv
mpirun --oversubscribe -np 10 ./scheduler --mode aipot --tasks 50000 --seed 42
cp logs/run_tasks.csv logs/aipot.csv

# 5. Analyze Results
python3 analyze_results.py logs/pot.csv logs/aipot.csv





