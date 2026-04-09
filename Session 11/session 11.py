import numpy as np

# Definición de funciones objetivo
def objective_given_function(position: np.ndarray) -> float:
    """
    Función del enunciado:
    f(x, y) = (x - 3.14)^2 + (y - 2.72)^2
              + sin(3x + 1.41) + sin(4y - 1.73)
    position: array de tamaño 2 -> [x, y]
    """
    x, y = position
    value = (x - 3.14) ** 2 + (y - 2.72) ** 2 \
            + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)
    return float(value)


def objective_second_function(position: np.ndarray) -> float:
    """
    Segunda función no lineal de ejemplo (puedes cambiarla si quieres):

    f(x, y) = x^2 + y^2 + 25*(sin(x)^2 + sin(y)^2)

    Tiene varios mínimos locales por los términos sinusoidales.
    """
    x, y = position
    value = x**2 + y**2 + 25*(np.sin(x)**2 + np.sin(y)**2)
    return float(value)

# Implementación básica de PSO
class Particle:
    def __init__(self, bounds, objective_func):
        """
        bounds: lista de tuplas [(x_min, x_max), (y_min, y_max)]
        objective_func: función a minimizar, recibe np.ndarray([x, y])
        """
        self.dim = len(bounds)
        self.position = np.array([
            np.random.uniform(low=b[0], high=b[1]) for b in bounds
        ], dtype=float)

        # Velocidad inicial pequeña
        self.velocity = np.zeros(self.dim, dtype=float)

        # Mejor posición personal y su valor
        self.best_position = self.position.copy()
        self.objective_func = objective_func
        self.best_value = self.objective_func(self.position)

    def update_velocity(self, global_best_position, w, c1, c2):
        """
        Actualiza la velocidad de la partícula según la ecuación de PSO:
        v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
        """
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)

        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)

        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        """
        Actualiza la posición: x = x + v
        Aplica límites de posición si se sale del espacio de búsqueda.
        """
        self.position = self.position + self.velocity

        # Aplicar límites
        for i, (low, high) in enumerate(bounds):
            if self.position[i] < low:
                self.position[i] = low
                self.velocity[i] = 0.0
            elif self.position[i] > high:
                self.position[i] = high
                self.velocity[i] = 0.0

        # Actualizar mejor posición personal si mejora
        current_value = self.objective_func(self.position)
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_position = self.position.copy()


def pso_optimize(objective_func,
                 bounds,
                 num_particles=30,
                 max_iters=100,
                 w=0.7,
                 c1=1.5,
                 c2=1.5,
                 verbose=True):
    """
    Ejecuta PSO para una función objetivo dada.

    objective_func: función a minimizar
    bounds: lista de tuplas [(x_min, x_max), (y_min, y_max)]
    num_particles: número de partículas
    max_iters: número máximo de iteraciones
    w, c1, c2: parámetros de PSO
    verbose: si True, imprime progreso cada ciertas iteraciones
    """
    # Inicializar partículas
    swarm = [Particle(bounds, objective_func) for _ in range(num_particles)]

    # Inicializar mejor global
    global_best_position = swarm[0].best_position.copy()
    global_best_value = swarm[0].best_value

    for p in swarm:
        if p.best_value < global_best_value:
            global_best_value = p.best_value
            global_best_position = p.best_position.copy()

    # Bucle principal
    for it in range(max_iters):
        for p in swarm:
            p.update_velocity(global_best_position, w, c1, c2)
            p.update_position(bounds)

            # Actualizar mejor global si mejora
            if p.best_value < global_best_value:
                global_best_value = p.best_value
                global_best_position = p.best_position.copy()

        if verbose and (it % 10 == 0 or it == max_iters - 1):
            print(f"Iteración {it+1}/{max_iters} | "
                  f"Mejor valor global: {global_best_value:.6f} | "
                  f"Mejor posición: {global_best_position}")

    return global_best_position, global_best_value

# Ejecución de ejemplo
def main():
    # Definir límites de búsqueda para x e y
    # Puedes ajustarlos según lo que quieras explorar
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]

    print("=== PSO sobre la función del enunciado ===")
    best_pos_1, best_val_1 = pso_optimize(
        objective_func=objective_given_function,
        bounds=bounds,
        num_particles=40,
        max_iters=150,
        w=0.7,
        c1=1.5,
        c2=1.5,
        verbose=True
    )
    print("\nResultado final (función enunciado):")
    print(f"  Mejor posición encontrada: {best_pos_1}")
    print(f"  Mejor valor encontrado: {best_val_1:.6f}")

    print("\n=== PSO sobre la segunda función no lineal ===")
    best_pos_2, best_val_2 = pso_optimize(
        objective_func=objective_second_function,
        bounds=bounds,
        num_particles=40,
        max_iters=150,
        w=0.7,
        c1=1.5,
        c2=1.5,
        verbose=True
    )
    print("\nResultado final (segunda función):")
    print(f"  Mejor posición encontrada: {best_pos_2}")
    print(f"  Mejor valor encontrado: {best_val_2:.6f}")


if __name__ == "__main__":
    main()