"""
Ant Colony Optimization (ACO) implementation for solving the Traveling Salesman Problem (TSP).
This script reads distance matrices and coordinates from CSV files, runs the ACO algorithm,
and outputs the best tour found along with visualizations and CSV results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

class ACO_TSP_CSV:
    """
    Ant Colony Optimization class for TSP, adapted to work with CSV input files.
    
    Attributes:
        dist_matrix (np.ndarray): Distance matrix between cities.
        n_cities (int): Number of cities.
        n_ants (int): Number of ants per iteration.
        n_iterations (int): Number of iterations to run.
        alpha (float): Pheromone importance factor.
        beta (float): Distance importance factor.
        rho (float): Pheromone evaporation rate.
        q (float): Pheromone deposit factor.
        pheromone (np.ndarray): Pheromone matrix.
        visibility (np.ndarray): Visibility matrix (inverse of distance).
    """
    def __init__(self, dist_matrix, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, rho=0.5, q=100):
        """
        Initialize the ACO algorithm with the given parameters.
        
        Args:
            dist_matrix (np.ndarray): Distance matrix for the TSP.
            n_ants (int): Number of ants.
            n_iterations (int): Number of iterations.
            alpha (float): Pheromone exponent.
            beta (float): Distance exponent.
            rho (float): Evaporation rate.
            q (float): Pheromone deposit constant.
        """
        self.dist_matrix = dist_matrix
        self.n_cities = dist_matrix.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Distance importance
        self.rho = rho      # Evaporation rate
        self.q = q          # Pheromone deposit factor
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((self.n_cities, self.n_cities)) / self.n_cities
        
        # Compute visibility matrix (inverse of distance, avoiding division by zero)
        self.visibility = 1.0 / (dist_matrix + np.eye(self.n_cities) * 1e-10)
        np.fill_diagonal(self.visibility, 0)

    def run(self):
        """
        Run the ACO algorithm for the specified number of iterations.
        
        Returns:
            tuple: (best_tour, best_length, history) where history is the best length per iteration.
        """
        best_tour = None
        best_length = float('inf')
        history = []

        for i in range(self.n_iterations):
            tours = self.generate_tours()
            lengths = [self.calculate_tour_length(tour) for tour in tours]
            
            min_idx = np.argmin(lengths)
            if lengths[min_idx] < best_length:
                best_length = lengths[min_idx]
                best_tour = tours[min_idx]
            
            history.append(best_length)
            self.update_pheromones(tours, lengths)
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Best Length: {best_length:.2f}")

        return best_tour, best_length, history

    def generate_tours(self):
        """
        Generate tours for all ants in the current iteration.
        
        Returns:
            list: List of tours, each tour is a list of city indices.
        """
        tours = []
        for _ in range(self.n_ants):
            tour = self.build_tour()
            tours.append(tour)
        return tours

    def build_tour(self):
        """
        Build a single tour starting from a random city and visiting all others.
        
        Returns:
            list: The constructed tour as a list of city indices.
        """
        start_city = np.random.randint(self.n_cities)
        tour = [start_city]
        unvisited = set(range(self.n_cities))
        unvisited.remove(start_city)

        while unvisited:
            current_city = tour[-1]
            probabilities = self.calculate_probabilities(current_city, unvisited)
            unvisited_list = list(unvisited)
            next_city = np.random.choice(unvisited_list, p=probabilities)
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return tour

    def calculate_probabilities(self, current_city, unvisited):
        """
        Calculate the probabilities of moving to each unvisited city from the current city.
        Args:
            current_city (int): Index of the current city.
            unvisited (set): Set of unvisited city indices.
        Returns:
            np.ndarray: Array of probabilities for each unvisited city.
        """
        unvisited_list = list(unvisited)
        pheromones = self.pheromone[current_city, unvisited_list] ** self.alpha
        visibilities = self.visibility[current_city, unvisited_list] ** self.beta
        
        probs = pheromones * visibilities
        sum_probs = np.sum(probs)
        
        if sum_probs == 0:
            return np.ones(len(unvisited_list)) / len(unvisited_list)
        
        return probs / sum_probs

    def calculate_tour_length(self, tour):
        """
        Calculate the total length of a given tour.
        
        Args:
            tour (list): List of city indices representing the tour.
        
        Returns:
            float: Total distance of the tour.
        """
        length = 0
        for i in range(len(tour)):
            length += self.dist_matrix[tour[i], tour[(i + 1) % self.n_cities]]
        return length

    def update_pheromones(self, tours, lengths):
        """
        Update the pheromone matrix based on the tours and their lengths.
        
        Args:
            tours (list): List of tours.
            lengths (list): Corresponding lengths of the tours.
        """
        self.pheromone *= (1 - self.rho)  # Evaporate pheromones
        for tour, length in zip(tours, lengths):
            contribution = self.q / length
            for i in range(len(tour)):
                c1, c2 = tour[i], tour[(i + 1) % self.n_cities]
                self.pheromone[c1, c2] += contribution
                self.pheromone[c2, c1] += contribution  # Symmetric matrix

def plot_tour(coords, tour, title, filename):
    """
    Plot the tour on a 2D plane and save the figure.
    
    Args:
        coords (np.ndarray): Coordinates of the cities.
        tour (list): List of city indices in the tour order.
        title (str): Title for the plot.
        filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(10, 7))
    tour_coords = coords[tour + [tour[0]]]  # Close the loop
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-', markersize=5, linewidth=1, color='blue')
    plt.scatter(coords[:, 0], coords[:, 1], color='red', zorder=5)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    """
    Main function to run ACO on multiple TSP datasets.
    Reads CSV files, runs the algorithm, plots results, and saves outputs.
    """
    datasets = [
        {'name': 'sp11', 'dist_csv': 'csv_datasets/sp11_dist.csv', 'xy_csv': 'csv_datasets/sp11_xy.csv'},
        {'name': 'sgb128', 'dist_csv': 'csv_datasets/sgb128_dist.csv', 'xy_csv': 'csv_datasets/sgb128_xy.csv'}
    ]

    for ds in datasets:
        print(f"\nProcessing {ds['name'].upper()} (CSV Input)...")
        
        # Read from CSV
        dist_matrix = pd.read_csv(ds['dist_csv'], header=None).values
        xy_coords = pd.read_csv(ds['xy_csv']).values
        
        # Set parameters based on dataset size
        if ds['name'] == 'sp11':
            n_ants, n_iter = 10, 50
        else:
            n_ants, n_iter = 30, 200
            
        aco = ACO_TSP_CSV(dist_matrix, n_ants=n_ants, n_iterations=n_iter)
        best_tour, best_length, history = aco.run()
        
        print(f"Best Length for {ds['name'].upper()}: {best_length:.2f}")
        
        # Visualization
        plot_tour(xy_coords, best_tour, f"TSP {ds['name'].upper()} - Length: {best_length:.2f}", f"{ds['name']}_tour_csv.png")
        
        # Write to CSV
        output_file = f"{ds['name']}_output.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', ds['name'].upper()])
            writer.writerow(['Best Length', best_length])
            writer.writerow(['Visit Order', 'City Index', 'X', 'Y'])
            for i, city_idx in enumerate(best_tour):
                writer.writerow([i + 1, city_idx, xy_coords[city_idx][0], xy_coords[city_idx][1]])
        
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Run the main function when the script is executed directly
    main()