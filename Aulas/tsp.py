from charles.charles import Population, Individual
from charles.search import hill_climb, sim_annealing
from copy import copy
from data.tsp_data import cities, distance_matrix
from charles.selection import fps, tournament_sel
from charles.mutation import swap_mutation
from charles.xo import cycle_xo

def get_fitness(self):
    """A simple objective function to calculate distances
    for the TSP problem.

    Returns:
        int: the total distance of the path
    """
    fitness = 0
    for i in range(len(self.representation)):
        # starting from the distance bw the last city and the first
        fitness += distance_matrix[self.representation[i-1]][self.representation[i]]
    return fitness


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

P = Population(size=20, optim="min", sol_size=len(cities),
                 valid_set=[i for i in range(len(cities))], repetition = False)

P.evolve(gens=100, xo_prob=0.9, mut_prob=0.15, select=tournament_sel,
         xo=cycle_xo, mutate=swap_mutation, elitism=True)



#hill_climb(pop)
#sim_annealing(pop)

