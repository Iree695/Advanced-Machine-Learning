"""
This script converts TSP (Traveling Salesman Problem) data files to CSV format.
It reads distance matrices and XY coordinates from text files and saves them as CSV.
"""
import pandas as pd
import numpy as np
import os

def load_dist_matrix(path, n_cities):
    """
    Loads the distance matrix from a text file.
    
    Args:
        path (str): Path to the file containing the distance matrix.
        n_cities (int): Number of cities in the TSP problem.
    
    Returns:
        np.ndarray: Distance matrix of shape (n_cities, n_cities).
    """
    distances = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            distances.extend(map(float, line.split()))
    return np.array(distances).reshape((n_cities, n_cities))

def load_xy_coords(path):
    """
    Loads the XY coordinates of the cities from a text file.
    
    Args:
        path (str): Path to the file containing the XY coordinates.
    
    Returns:
        np.ndarray: Array of coordinates of shape (n_cities, 2) with X and Y columns.
    """
    coords = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 3:
                coords.append([float(parts[1]), float(parts[2])])
            elif len(parts) == 2:
                coords.append([float(parts[0]), float(parts[1])])
    return np.array(coords)

def convert_to_csv():
    """
    Converts the specified TSP datasets to CSV format.
    Creates a 'csv_datasets' folder and saves the distance matrices and coordinates as CSV files.
    """
    datasets = [
        {'name': 'sp11', 'n': 11},
        {'name': 'sgb128', 'n': 128}
    ]
    
    os.makedirs('csv_datasets', exist_ok=True)
    
    for ds in datasets:
        # Convert distances
        dist_matrix = load_dist_matrix(f"tsp_data/{ds['name']}_dist.txt", ds['n'])
        df_dist = pd.DataFrame(dist_matrix)
        df_dist.to_csv(f"csv_datasets/{ds['name']}_dist.csv", index=False, header=False)
        
        # Convert coordinates
        xy_coords = load_xy_coords(f"tsp_data/{ds['name']}_xy.txt")
        df_xy = pd.DataFrame(xy_coords, columns=['X', 'Y'])
        df_xy.to_csv(f"csv_datasets/{ds['name']}_xy.csv", index=False)
        
        print(f"Converted {ds['name']} to CSV in the 'csv_datasets' folder.")

if __name__ == "__main__":
    # Run the conversion when the script is executed directly
    convert_to_csv()