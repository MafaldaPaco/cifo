from charles.charles import Population, Individual
from charles.selection import fps, tournament_sel, sigma_scaling_selection, rank_selection
from charles.mutation import swap_mutation, inversion_mutation, binary_mutation, shuffle_mutation
from charles.crossover import order_crossover, single_point_xo, position_based_crossover, geometric_xo
from copy import copy
from data.data import vrp_data as data, locations
import numpy as np
import matplotlib.pyplot as plt


def get_fitness(self):
    """A simple objective function to calculate distances for the VRP problem, with a penalization for excessive use of vehicles.
    Returns:
        int: the total distance of the route
    """
    fitness = 0
    vehicles_used = sum(1 for route in self.representation if route)
    visited_locations = set()

    # Looping over each vehicle's route
    for route in self.representation:
        if not route:  
            continue
        # Adding the distance from the depot to the first location
        fitness += data['distance_matrix'][data['depot']][route[0]]
        # Adding the distances between consecutive locations in the route
        for i in range(len(route) - 1):
            fitness += data['distance_matrix'][route[i]][route[i + 1]]
        # Adding the distance from the last location back to the depot
        fitness += data['distance_matrix'][route[-1]][data['depot']]
        # Updating visited locations
        visited_locations.update(route)
    
    # Penalizing for use of more vehicles than necessary
    penalty = 0
    if vehicles_used > 1:
        penalty = (vehicles_used-1) * 500
    
    # Penalizing for not visiting all locations
    unvisited = set(range(1, len(data['distance_matrix']))) - visited_locations
    penalty += len(unvisited) * 1000  
    
    return fitness + penalty

def get_neighbours(self):
    """A neighbourhood function for the TSP problem. Switch
    indexes around in pairs.

    Returns:
        list: a list of individuals
    """
    n = [copy(self.representation) for _ in range(len(self.representation)-1)]

    for i, ne in enumerate(n):
        ne[i], ne[i+1] = ne[i+1], ne[i]

    n = [Individual(ne) for ne in n]
    return n


# Monkey patching
Individual.get_fitness = get_fitness
Individual.get_neighbours = get_neighbours


P = Population(size=20, optim="min", sol_size=len(locations)-1,
                 valid_set=[i for i in range(1, len(locations))], num_vehicles=data['num_vehicles'])

res= P.evolve(gens=100, xo_prob=0.9, mut_prob=0.15, select=rank_selection,
         xo=position_based_crossover, mutate=swap_mutation, elitism=1)
print(res)
