from charles.charles import Population, Individual
from copy import copy
from data import data


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

def get_fitness(self):
    """A simple objective function to calculate distances for the VRP problem.
    Returns:
        int: the total distance of the routes
    """
    fitness = 0
    # Loop over each vehicle's route
    for route in self.representation:
        if not route:  
            continue
        # Add the distance from the depot to the first location
        fitness += data['distance_matrix'][data['depot']][route[0]]
        # Add the distances between consecutive locations in the route
        for i in range(len(route) - 1):
            fitness += data['distance_matrix'][route[i]][route[i + 1]]
        # Add the distance from the last location back to the depot
        fitness += data['distance_matrix'][route[-1]][data['depot']]
        
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

'''P = Population(size=20, optim="min", sol_size=len(cities),
                 valid_set=[i for i in range(len(cities))], repetition = False)

P.evolve(gens=100, xo_prob=0.9, mut_prob=0.15, select=tournament_sel,#
         xo=cycle_xo, mutate=swap_mutation, elitism=True)
'''

#hill_climb(pop)
#sim_annealing(pop)


print()
a= [[] for _ in range(data['num_vehicles'])]
fitness = 0
# Loop over each vehicle's route
for route in [[] for _ in range(data['num_vehicles'])]:
    print(data['distance_matrix'][data['depot']])
    print([route])
    print([np.random.uniform(size=size[i]) for i in range(len(size))])
    if not route:  
        print('b')
        continue
    # Add the distance from the depot to the first location
    fitness += data['distance_matrix'][data['depot']][route[0]]
    print(f"{data['distance_matrix'][data['depot']][route[0]]}, ahahah")
    # Add the distances between consecutive locations in the route
    for i in range(len(route) - 1):
        fitness += data['distance_matrix'][route[i]][route[i + 1]]
    # Add the distance from the last location back to the depot
    fitness += data['distance_matrix'][route[-1]][data['depot']]