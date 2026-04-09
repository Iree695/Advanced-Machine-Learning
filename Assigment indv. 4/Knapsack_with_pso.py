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

# Class Particle
# - Position: binary vector of length N_items
# - Speed: real vector
# - Personal best: looks for the max

class Particle:
    def __init__(self, n_items):
        # Position
        position = []
        for _ in range(n_items):
            position.append(random.randint(0, 1))
        self.position = np.array(position, dtype = float)

        # Speed
        speed = []
        for _ in range(n_items):
            speed.append(random.uniform(-1, 1))
        self.speed = np.array(speed) 

        # Personal Best --> Max means that it starts from -inf
        self.personal_best_position = self.position.copy()
        self.personal_best_value = float("-inf")

    def evaluate(self, weight, values, capacity):
        """With this function we can evaluate the current position and  update the personal best"""
        # For evaluating the current position
        evalue = fitness(self.position, self.weight, self.values, self.capacity)

        # For pdating the personal best
        if evalue > self.personal_best_value:
            self.personal_best_value = evalue
            self.personal_best_position = self.position.copy()
        else:
            return evalue
        
# Knapsack Class (Main)
# - Maximization Problem
# - Speed Formula: V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X) 
    # where (pbest - X) and (gbest - X) are bit differencies
    # result in {-1, 0, +1}  pushing speed towards where the best bits are
# - Binary update of the position using sigmoid
    # Here we convert the V in a probability with the sigmoid function. Bits are decided randomly
    # So:
        # prob[i] = σ(V[i])
        # X[i] = 1 if random() < prob[i]
        # X[i] = 0 otherwise

class Knapsack:
    def __init__(self, weights, values, capacity, n_particles, max_iteration, 
                 w = 0.7, c1 = 1.5, c2 = 1.5):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n_particles = n_particles
        self.max_iteration = max_iteration

        # Hyperparameters:
        self.w = w   # Inertia:memory of the previous speed
        self.c1 = c1 # Cognitive: attraction to one's own best
        self.c2 = c2 # Social: attraction towards the globals best

        n_items = len(weights)
        self.swarm = []
        for _ in range(n_particles):
            self.swarm.append(Particle(n_items))

        # Global best 
        self.global_best_pose = self.swarm[0].position.copy()
        self.global_best_value = float("inf")

    def optimize(self):
        for iteration in range(self.max_iterations):
            # Evaluation and updating the global best:
            for particle in self.swarm:
                value = particle.evaluate(self.weights, self.values, self.capacity)

                if value > self.global_best_value:
                    self.global_best_value = value
                    self.global_best_pose = particle.position.copy()

            # Updating speed and position:
            for particle in self.swarm:
                r1 = random.random()
                r2 = random.random()
                # Speed Formula
                cognitive = self.c1 * r1 * (particle.personal_best_position - particle.position)
                social = self.c2 * r2 * (particle.global_best_position - particle.position)

                particle_speed = self.w * particle_speed + cognitive + social
            
            # Binary updating the position
            prob = sigmoid(particle_speed)
            position = []
            for p in prob:
                if random.random() < p:
                    position.append(1)
                else:
                    position.append(0)
            particle.position = np.array(position, dtype = float)


