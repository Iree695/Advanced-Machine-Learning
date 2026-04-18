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

        # Initialize pheromones
        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities

        # Compute visibility ( inverse of distance )
        self.visibility = 1/ (dist_matrix + np.eye(self.n_cities) * 1e-10)
        np.fill_diagonal(self.visibility, 0)

    def run(self):
        best_tour = None
        best_length = float("inf")
        history = []

        for iteration in range(self.n_iterations):
            tours = self.generate_tours()
            lenghts = []
            for tour in tours:
                lenght = self.tour_lenght(tour)
                lenghts.append(lenght)

                min_idx = np.argmin(lenghts)
                if lenghts[min_idx] < best_length:
                    best_length = lenghts[min_idx]
                    best_tour = tours[min_idx]
                
                history.append(best_length)
                self.update_pheromones(tours, lenghts)

                if iteration % 10 == 0:
                    print(f"Iteration {iteration}, Best Length: {best_length}")

            return best_tour, best_length, history
        
            