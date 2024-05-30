from operator import attrgetter
from random import sample, uniform, choice
import numpy as np

def select(population):
    tournament = sample(population.individuals, k=3)
    if population.optim == "max":
        return max(tournament, key=attrgetter('fitness'))
    else:
        return min(tournament, key=attrgetter('fitness'))
    


def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """
    if population.optim == "max":
        total_fitness = sum([i.fitness for i in population])
        r = uniform(0, total_fitness)
        position = 0
        for individual in population:
            position += individual.fitness
            if position > r:
                return individual
    elif population.optim == "min":
        raise NotImplementedError
    else:
        raise Exception(f"Optimization not specified (max/min)")


def tournament_sel(population, tour_size=3):
    tournament = [choice(population) for _ in range(tour_size)]
    if population.optim == "max":
        return max(tournament, key=attrgetter('fitness'))
    elif population.optim == "min":
        return min(tournament, key=attrgetter('fitness'))


def sigma_scaling(population):
    """Sigma sclaing selection implementation. Reference: Mitchell, M. (1996). An introduction to genetic algorithms. https://doi.org/10.7551/mitpress/3927.001.0001 

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """
    fitnesses = [ind.fitness for ind in population.individuals]
    mean = np.mean(fitnesses)
    sigma = np.std(fitnesses)

    #Under sigma scaling, an individual's expected value is a function of its fitness, the population mean, and the population standard deviation
    if sigma == 0:
        scaled_fitnesses = [1 for _ in fitnesses]
    else:
        scaled_fitnesses = [1 + (fitness - mean) / (2 * sigma) for fitness in fitnesses]

    # Avoiding fitnesses of 0, so that individuals with very low fitness have some small chance of reproducing
    scaled_fitnesses = [max(f, 0.1) for f in scaled_fitnesses]

    total_scaled_fitness = sum(scaled_fitnesses)
    r = np.random.uniform(0, total_scaled_fitness)
    position = 0
    # Select individual based on cumulative probability - it iterates through the population to find the individual whose cumulative probability exceeds the random value r
    for individual, scaled_fitness in zip(population.individuals, scaled_fitnesses):
        position += scaled_fitness
        if position > r:
            return individual
