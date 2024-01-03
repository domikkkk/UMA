import random
import numpy as np
import random
import cec2017

from cec2017.functions import f1


UPPER_BOUND = 100
DIMENSIONALITY = 2  # długość tylko 2, 10, 20, 30, 50 lub 100


class Point:
    def __init__(self, array=None) -> None:
        self.array = array
        if array is None:
            self.array = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(DIMENSIONALITY))

    def mutation(self, sigma: float):
        self.array = np.array([gen + sigma * random.gauss(0, 1) for gen in self.array])
        return

    def copy(self):
        return Point(self.array[:])


class EvolutionAlgorithm:
    def __init__(self, population, function, population_size=1000, tournament_size=2, sigma=0.1, steps=100) -> None:
        self.population = population
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.sigma = sigma
        self.steps = steps
        self.func = function
        self.mean, self.std = self.mean_and_deviation()

    def eval_func(self, point: Point):
        return self.func(point.array)

    def set_new_p_size(self, delta_size, percent=0):
        if percent:
            delta_size = int(self.population_size * delta_size // 100)
        self.population_size = self.population_size + delta_size
        if delta_size < 0:
            population = sorted(self.population, key=self.eval_func)
            self.population = population[:self.population_size]
        elif delta_size > 0:
            self.population = self.population + [Point() for _ in range(delta_size)]
        return

    def set_new_sigma(self, delta_sigma, percent=0):
        if percent:
            delta_sigma = int(self.sigma * delta_sigma // 100)
        self.sigma = self.sigma + delta_sigma
        return

    def tournament_selection(self) -> Point:
        new_points = [random.choice(self.population) for _ in range(self.tournament_size)]
        best_point = sorted(new_points, key=self.eval_func)[0].copy()
        return best_point

    def tournament_for_all(self):
        new_population = np.array([])
        for _ in range(self.population_size):
            point = self.tournament_selection()
            new_population = np.append(new_population, point)
        self.population = np.array(new_population, copy=True)

    def mutate_all(self):
        for point in self.population:
            point.mutation(self.sigma)
        return

    def step(self):
        self.tournament_for_all()
        # można dodać krzyżowanie
        self.mutate_all()
        self.steps = self.steps - 1
        mean, std = self.mean_and_deviation()
        return mean

    def mean_and_deviation(self):
        array = np.array([self.eval_func(p) for p in self.population])
        mean = np.mean(array)
        std = np.std(array)
        return mean, std
            
    def __str__(self):
        return f"{len(self.population)}\n{[list(point.array) for point in self.population]}"


points = np.array([Point() for _ in range(10)])
e = EvolutionAlgorithm(points, f1, 10, sigma=5)
print(e.mean)
print(e.step())
