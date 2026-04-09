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
    
# Sigmoid function -- Key piece for the  Binary PSO
# - PROBLEM: Position =  [0,1] while speed is a real number. It is imposible to do directly x = X + V
# - SOLUTION: Changing the speed to a probability with the sigmoid function and then choose randomly
# - Ecuation of the sigmoid function --> σ(v) = 1 / (1 + e^(-v)) 
#   - If v is very positive --> σ(v) = 1 --> bit = 1
#   - If v is very negative --> σ(v) = 0 --> bit = 0

def sigmoid(v):
    func = 1 / 1 + np.exp(-np.clip(v, -15, 15))
    return func