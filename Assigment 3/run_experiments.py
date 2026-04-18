import numpy as np
import matplotlib.pyplot as plt
from pso_implementation import GlobalBestPSO, VonNeumannPSO, DynamicNeighborhoodPSO, rastrigin, ackley
import json
import os

def run_experiment(pso_class, func, dim, bounds, num_particles, max_iter, runs=10, **kwargs):
    all_histories = []
    best_vals = []
    for _ in range(runs):
        pso = pso_class(func, dim, bounds, num_particles, max_iter, **kwargs)
        _, best_val = pso.optimize()
        all_histories.append(pso.history)
        best_vals.append(best_val)
    
    avg_history = np.mean(all_histories, axis=0)
    return avg_history, np.mean(best_vals), np.std(best_vals)

# Configuration
DIM = 10
BOUNDS = (-5.12, 5.12)
NUM_PARTICLES = 36 # Square number for Von Neumann grid
MAX_ITER = 100
RUNS = 10

results = {}

for func_name, func in [("Rastrigin", rastrigin), ("Ackley", ackley)]:
    print(f"Running experiments for {func_name}...")
    results[func_name] = {}
    
    # Global Best
    avg_hist, mean_best, std_best = run_experiment(GlobalBestPSO, func, DIM, BOUNDS, NUM_PARTICLES, MAX_ITER, runs=RUNS)
    results[func_name]["GlobalBest"] = {"history": avg_hist.tolist(), "mean": mean_best, "std": std_best}
    
    # Von Neumann
    avg_hist, mean_best, std_best = run_experiment(VonNeumannPSO, func, DIM, BOUNDS, NUM_PARTICLES, MAX_ITER, runs=RUNS)
    results[func_name]["VonNeumann"] = {"history": avg_hist.tolist(), "mean": mean_best, "std": std_best}
    
    # Dynamic
    avg_hist, mean_best, std_best = run_experiment(DynamicNeighborhoodPSO, func, DIM, BOUNDS, NUM_PARTICLES, MAX_ITER, runs=RUNS)
    results[func_name]["Dynamic"] = {"history": avg_hist.tolist(), "mean": mean_best, "std": std_best}

    # Save results
    with open(os.path.join(os.path.dirname(__file__), "experiment_results.json"), "w") as f:
        json.dump(results, f)

    # Generate plot for this function
    plt.figure(figsize=(10, 6))
    for pso_type in results[func_name]:
        history = results[func_name][pso_type]["history"]
        plt.plot(history, label=f"{pso_type} (Mean: {results[func_name][pso_type]['mean']:.4f})")
    
    plt.title(f"PSO Convergence on {func_name} Function")
    plt.xlabel("Iterations")
    plt.ylabel("Best Value Found (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    filename = os.path.join(os.path.dirname(__file__), f"convergence_{func_name.lower()}.png")
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

    print("\nExperiments completed successfully.")
