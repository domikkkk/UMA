#Autor: Dominik Sidorczuk, Tomasz Sroka
import random
import numpy as np
import random
import cec2017

from cec2017.functions import f1

from typing import Tuple
from environment_base import Environment


UPPER_BOUND = 100
DIMENSIONALITY = 2  # długość tylko 2, 10, 20, 30, 50 lub 100


class Point:
    def __init__(self, array=None) -> None:
        self.array = array
        if array is None:
            self.array = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(DIMENSIONALITY))

    def mutation(self, sigma: float) -> np.array:
        self.array = np.array([gen + sigma * random.gauss(0, 1) for gen in self.array])
        return

    def copy(self):
        return Point(self.array[:])
    
    def __repr__(self):
        return ",".join(str(gen) for gen in self.array)


class EvolutionAlgorithm(Environment):
    def __init__(self, population, function, tournament_size=2, sigma=0.1, steps=100) -> None:
        self.population = population
        self._population_size = len(population) 
        self.tournament_size = tournament_size
        self._sigma = sigma
        self.steps = steps
        self.func = function

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self,value):
        self._population_size==min(0,value)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self,value):
        self._sigma=min(0,value)

    def eval_func(self, point: Point) -> np.ndarray[float]:
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

    def set_new_sigma(self, delta_sigma, percent=0):
        if percent:
            delta_sigma = int(self.sigma * delta_sigma // 100)
        self.sigma = self.sigma + delta_sigma

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

    def step(self) -> float:
        self.tournament_for_all()
        # można dodać krzyżowanie
        self.mutate_all()
        self.steps = self.steps - 1
        mean, std = self.mean_and_deviation()
        return mean

    def mean_and_deviation(self) -> Tuple[float,float]:
        array = np.array([self.eval_func(p) for p in self.population])
        mean = np.mean(array)
        std = np.std(array)
        return mean, std
            
    def __str__(self):
        return f"{len(self.population)}\n{[list(point.array) for point in self.population]}"
