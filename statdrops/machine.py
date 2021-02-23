import random
import numpy as np

def population_creator(diameter_class, num_drops):

    population = []
    for i in range(len(diameter_class)):
        n = 0
        while n < num_drops[i]:
            population.append(diameter_class[i])
            n += 1

    return population


def observation_generator(population, num_obervations):
    return np.random.choice(population, num_obervations, replace=True)
