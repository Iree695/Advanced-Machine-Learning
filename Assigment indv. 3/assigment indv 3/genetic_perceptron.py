import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import time
import os

# DIRECTORY
directorio_del_script = os.path.dirname(os.path.abspath(__file__))
ruta_grafico = os.path.join(directorio_del_script, "convergence_plot.png")

# MODEL DEFINITION (From Hill Climbing)
class Perceptron:
    def __init__(self, n_features, weights=None):
        if weights is not None:
            self.w = weights
        else:
            self.w = np.random.randn(n_features)

    def predict(self, X):
        return np.where(np.dot(X, self.w) >= 0, 1, -1) # only gives -1 or 1 no 0

# FITNESS FUNCTION
def fitness_function(weights, X, y):
    model = Perceptron(X.shape[1], weights=weights)
    y_pred = model.predict(X)
    # Using F1-macro as in the reference code
    return f1_score(y, y_pred, average='macro')

# GENETIC ALGORITHM COMPONENTS

# Initialization
def initialize_population(pop_size, n_features):
    """One chosen initialization function: Random Gaussian"""
    return [np.random.randn(n_features) for _ in range(pop_size)]

# Selection 
def tournament_selection(population, fitnesses, k=3):
    """One selection algorithm: Tournament Selection"""
    selected = []
    for _ in range(len(population)):
        indices = np.random.choice(len(population), k)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        selected.append(population[best_idx])
    return selected

# Crossover 
def crossover_one_point(parent1, parent2):
    """Crossover 1: One-point recombination"""
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def crossover_uniform(parent1, parent2):
    """Crossover 2: Uniform recombination"""
    mask = np.random.rand(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

# Mutation
def mutation_gaussian(individual, mutation_rate=0.1, sigma=0.1):
    """Mutation 1: Gaussian noise (Real-valued equivalent of bit-flip)"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[i] += np.random.randn() * sigma
    return mutated

def mutation_uniform(individual, mutation_rate=0.1):
    """Mutation 2: Uniform mutation (Reemplaza un peso por uno nuevo aleatorio)"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[i] = np.random.randn()  # Genera un nuevo peso desde cero
    return mutated

# 4. Main Genetic Algorithm
def genetic_algorithm(X, y, n_features, pop_size=50, n_generations=100, mutation_rate=0.1):
    population = initialize_population(pop_size, n_features)
    history = []
    
    best_overall_w = None
    best_overall_fitness = -1

    for gen in range(n_generations):
        # Evaluation
        fitnesses = [fitness_function(ind, X, y) for ind in population]
        
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_overall_fitness:
            best_overall_fitness = fitnesses[current_best_idx]
            best_overall_w = population[current_best_idx].copy()
        
        history.append(best_overall_fitness)

        # Selection
        parents = tournament_selection(population, fitnesses)

        # Variation (Crossover + Mutation)
        new_population = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[(i+1)%pop_size]
            
            # Randomly choose between two crossover methods
            if np.random.rand() < 0.5:
                c1, c2 = crossover_one_point(p1, p2)
            else:
                c1, c2 = crossover_uniform(p1, p2)
            
            # Randomly choose between two mutation methods
            if np.random.rand() < 0.5:
                c1 = mutation_gaussian(c1, mutation_rate)
                c2 = mutation_gaussian(c2, mutation_rate)
            else:
                c1 = mutation_uniform(c1, mutation_rate)
                c2 = mutation_uniform(c2, mutation_rate)
                
            new_population.extend([c1, c2])
        
        population = new_population[:pop_size]
        
        if (gen + 1) % 10 == 0:
            print(f"Generation {gen+1}: Best Fitness = {best_overall_fitness:.4f}")

    return best_overall_w, history

# VANILLA PERCEPTRON 
def train_vanilla_perceptron(X, y, lr=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    
    # Map y from {-1, 1} to {0, 1} for standard update if needed, 
    # but here we use the classic sign-based update.
    start_time = time.time()
    for _ in range(epochs):
        for idx, x_i in enumerate(X):
            linear_output = np.dot(x_i, weights)
            y_predicted = np.sign(linear_output)
            
            # Perceptron update rule: w = w + lr * (y - y_pred) * x
            if y[idx] != y_predicted:
                weights += lr * (y[idx] - y_predicted) * x_i
    end_time = time.time()
    
    return weights, end_time - start_time

# EXPERIMENTING WITH THE WISCONSIN CANCER DATASET
if __name__ == "__main__":
    # Load and preprocess data
    data = load_breast_cancer()
    X, y = data.data, data.target
    # Convert y from {0, 1} to {-1, 1} for Perceptron compatibility
    y = np.where(y == 0, -1, 1)
    
    # Add bias term
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    
    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.3, random_state=42)
    
    # Scale features (important for weight-based optimization)
    scaler = StandardScaler()
    X_train[:, 1:] = scaler.fit_transform(X_train[:, 1:])
    X_test[:, 1:] = scaler.transform(X_test[:, 1:])

    print("Training Genetic Algorithm Perceptron")
    start_ga = time.time()
    best_w_ga, history_ga = genetic_algorithm(X_train, y_train, X_train.shape[1], pop_size=50, n_generations=150)
    end_ga = time.time()
    time_ga = end_ga - start_ga

    print("\n Training Vanilla Perceptron ")
    best_w_vanilla, time_vanilla = train_vanilla_perceptron(X_train, y_train, epochs=150)

    # Evaluation
    model_ga = Perceptron(X_train.shape[1], weights=best_w_ga)
    model_vanilla = Perceptron(X_train.shape[1], weights=best_w_vanilla)
    
    y_pred_ga = model_ga.predict(X_test)
    y_pred_vanilla = model_vanilla.predict(X_test)
    
    acc_ga = accuracy_score(y_test, y_pred_ga)
    acc_vanilla = accuracy_score(y_test, y_pred_vanilla)
    
    f1_ga = f1_score(y_test, y_pred_ga, average='macro')
    f1_vanilla = f1_score(y_test, y_pred_vanilla, average='macro')

    print("RESULTS COMPARISON")
    print(f"{'Metric':<15} | {'GA Perceptron':<15} | {'Vanilla Perceptron':<15}")
    print(f"{'Accuracy':<15} | {acc_ga:<15.4f} | {acc_vanilla:<15.4f}")
    print(f"{'F1-Score':<15} | {f1_ga:<15.4f} | {f1_vanilla:<15.4f}")
    print(f"{'Time (s)':<15} | {time_ga:<15.4f} | {time_vanilla:<15.4f}")
    
    # Plotting convergence for GA
    plt.figure(figsize=(10, 6))
    plt.plot(history_ga, label="GA Fitness (Best)")
    plt.axhline(y=f1_vanilla, color='r', linestyle='--', label="Vanilla Perceptron F1")
    plt.xlabel("Generation")
    plt.ylabel("F1-Score (Macro)")
    plt.title("GA Perceptron Convergence vs Vanilla Perceptron Baseline")
    plt.legend()
    plt.grid(True)
    plt.savefig(ruta_grafico)
    print("\nConvergence plot saved")