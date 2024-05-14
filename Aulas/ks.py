from operator import attrgetter
from charles.charles import Population, Individual
from copy import copy
from data.ks_data import values, weights, capacity
import numpy as np
from charles.selection import fps, tournament_sel
from charles.mutation import binary_mutation
from charles.xo import single_point_xo
from random import random
from charles.search import hill_climb, sim_annealing


def get_fitness(self):
    """A function to calculate the total value of the bag if the capacity is not exceeded.
    If the capacity is exceeded, it will return a negative fitness specifying how bad the solution is
    (how far the solution is from the capacity)
    Returns:
        int: Total value
    """
    # Alternative Method 1 to calculate fitness :
    # (for total weight, the exact same thing can be done with the list weights instead of values)

    # fitness = 0
    # for bit in range(len(self.representation)):
    #     if self.representation[bit] == 1:
    #         fitness += values[bit]

    # Alternative Method 2:
    # fitness = sum([self.representation[bit]*values[bit] for i in range(len(self.representation))])

    total_w = np.dot(self.representation, weights)
    if total_w < capacity:
        return np.dot(self.representation, values)
    else:
        return capacity - total_w


def get_neighbours(self):
    """A neighbourhood function for the knapsack problem,
    for each neighbour, flips the bits
    Returns:
        list: a list of individuals
    """
    # n -> neighbourhood
    n = [copy(self.representation) for _ in range(len(self.representation))]

    for i, ne in enumerate(n):
        if ne[i] == 1:
            ne[i] = 0
        elif ne[i] == 0:
            ne[i] = 1

    n = [Individual(ne) for ne in n]
    return n


# Monkey patching
Individual.get_fitness = get_fitness
Individual.get_neighbours = get_neighbours

# init P with N indvs
P = Population(size=20, optim="max", sol_size=len(values),
               valid_set=[0, 1], repetition=True)

P.evolve(gens=100, xo_prob=0.9, mut_prob=0.15, select=fps,
         xo=single_point_xo, mutate=binary_mutation, elitism=True)


#hill_climb(pop)
#sim_annealing(pop)