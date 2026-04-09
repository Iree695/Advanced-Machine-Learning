import numpy as np
import random

# Generate items
# - A group of items with different values(€) and weights(kg)
# - They are being generated randomly, there isn't a fixed seed
def generate_items(n_items=10):
    weights = np.round(np.random.uniform(1, 10, size = n_items), 1)
    values = np.round(np.random.uniform(10, 100, size = n_items), 0)
    return weights, values


