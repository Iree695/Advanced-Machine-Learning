import numpy as np
import random

# Generate items
# - A group of items with different values(€) and weights(kg)
# - They are being generated randomly, there isn't a fixed seed
def generate_items(n_items=10):
    weights = np.round(np.random.uniform(1, 10, size = n_items), 1)
    values = np.round(np.random.uniform(10, 100, size = n_items), 0)
    return weights, values

# Fitness function
# - If we exceed the weight → fitness = 0 so instead of minimazing the solution we maximize it.

def fitness(pose, weights, values, capacity):
    total_weight = np.dot(pose, weights)
    total_values = np.dot(pose, values)

    if total_weight > capacity:
        return 0
    else:
        return total_values