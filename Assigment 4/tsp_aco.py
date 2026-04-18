import numpy as np
import matplotlib.pyplot as plt
import os

# Class for Ant Colony Optimization:
class ACO:
    def __init__(self, dist_matrix, n_ants =20, n_iterations = 100, alpha = 1, beta = 2, rho = 0.5, q = 100):
        self.dist_matrix = dist_matrix
        self.n_cities = dist_matrix.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha # --> pheromone importance
        self.beta = beta # --> distance importance
        self.rho = rho # --> evaporation rate
        self.q = q # --> pheromone deposit factor