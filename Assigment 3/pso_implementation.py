import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.pbest_pos = np.copy(self.position)
        self.pbest_val = float('inf')
        self.current_val = float('inf')

class PSOBase:
    def __init__(self, func, dim, bounds, num_particles, max_iter, w=0.729, c1=1.494, c2=1.494):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]
        self.gbest_pos = None
        self.gbest_val = float('inf')
        self.history = []

    def update_particle(self, particle, lbest_pos):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        particle.velocity = (self.w * particle.velocity + 
                             self.c1 * r1 * (particle.pbest_pos - particle.position) + 
                             self.c2 * r2 * (lbest_pos - particle.position))
        particle.position += particle.velocity
        # Boundary check
        particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

    def optimize(self):
        raise NotImplementedError("Subclasses must implement optimize method")

class GlobalBestPSO(PSOBase):
    def optimize(self):
        for i in range(self.max_iter):
            for p in self.particles:
                p.current_val = self.func(p.position)
                if p.current_val < p.pbest_val:
                    p.pbest_val = p.current_val
                    p.pbest_pos = np.copy(p.position)
                
                if p.current_val < self.gbest_val:
                    self.gbest_val = p.current_val
                    self.gbest_pos = np.copy(p.position)
            
            for p in self.particles:
                self.update_particle(p, self.gbest_pos)
            
            self.history.append(self.gbest_val)
        return self.gbest_pos, self.gbest_val

# Test functions
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

class VonNeumannPSO(PSOBase):
    def __init__(self, func, dim, bounds, num_particles, max_iter, grid_size=None, **kwargs):
        super().__init__(func, dim, bounds, num_particles, max_iter, **kwargs)
        if grid_size is None:
            # Assume square grid
            side = int(np.sqrt(num_particles))
            self.grid_rows = side
            self.grid_cols = num_particles // side
        else:
            self.grid_rows, self.grid_cols = grid_size
        
        # Ensure we have enough particles for the grid
        self.num_particles = self.grid_rows * self.grid_cols
        self.particles = [Particle(dim, bounds) for _ in range(self.num_particles)]

    def get_neighbors(self, idx):
        r, c = divmod(idx, self.grid_cols)
        neighbors = []
        # Von Neumann neighbors (up, down, left, right) with wrap-around
        neighbors.append(((r - 1) % self.grid_rows) * self.grid_cols + c)
        neighbors.append(((r + 1) % self.grid_rows) * self.grid_cols + c)
        neighbors.append(r * self.grid_cols + (c - 1) % self.grid_cols)
        neighbors.append(r * self.grid_cols + (c + 1) % self.grid_cols)
        # Include the particle itself
        neighbors.append(idx)
        return neighbors

    def optimize(self):
        for i in range(self.max_iter):
            # Evaluate all particles
            for p in self.particles:
                p.current_val = self.func(p.position)
                if p.current_val < p.pbest_val:
                    p.pbest_val = p.current_val
                    p.pbest_pos = np.copy(p.position)
                
                if p.current_val < self.gbest_val:
                    self.gbest_val = p.current_val
                    self.gbest_pos = np.copy(p.position)

            # Update each particle based on its local neighborhood best
            for idx, p in enumerate(self.particles):
                neighbor_indices = self.get_neighbors(idx)
                lbest_val = float('inf')
                lbest_pos = None
                for n_idx in neighbor_indices:
                    neighbor = self.particles[n_idx]
                    if neighbor.pbest_val < lbest_val:
                        lbest_val = neighbor.pbest_val
                        lbest_pos = np.copy(neighbor.pbest_pos)
                
                self.update_particle(p, lbest_pos)
            
            self.history.append(self.gbest_val)
        return self.gbest_pos, self.gbest_val

class DynamicNeighborhoodPSO(PSOBase):
    def __init__(self, func, dim, bounds, num_particles, max_iter, regroup_period=10, **kwargs):
        super().__init__(func, dim, bounds, num_particles, max_iter, **kwargs)
        self.regroup_period = regroup_period
        self.neighbors = self.generate_random_neighbors()

    def generate_random_neighbors(self):
        # Each particle gets a random set of neighbors (e.g., 4 random particles)
        neighbors = {}
        for i in range(self.num_particles):
            # Randomly pick 4 other particles as neighbors
            n_indices = np.random.choice(self.num_particles, 4, replace=False).tolist()
            if i not in n_indices:
                n_indices.append(i)
            neighbors[i] = n_indices
        return neighbors

    def optimize(self):
        for i in range(self.max_iter):
            # Regroup neighbors periodically
            if i % self.regroup_period == 0:
                self.neighbors = self.generate_random_neighbors()

            # Evaluate all particles
            for p in self.particles:
                p.current_val = self.func(p.position)
                if p.current_val < p.pbest_val:
                    p.pbest_val = p.current_val
                    p.pbest_pos = np.copy(p.position)
                
                if p.current_val < self.gbest_val:
                    self.gbest_val = p.current_val
                    self.gbest_pos = np.copy(p.position)

            # Update each particle based on its dynamic neighborhood best
            for idx, p in enumerate(self.particles):
                neighbor_indices = self.neighbors[idx]
                lbest_val = float('inf')
                lbest_pos = None
                for n_idx in neighbor_indices:
                    neighbor = self.particles[n_idx]
                    if neighbor.pbest_val < lbest_val:
                        lbest_val = neighbor.pbest_val
                        lbest_pos = np.copy(neighbor.pbest_pos)
                
                self.update_particle(p, lbest_pos)
            
            self.history.append(self.gbest_val)
        return self.gbest_pos, self.gbest_val