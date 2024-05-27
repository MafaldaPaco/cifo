from charles.charles import Population, Individual
from copy import copy
from data.data import vrp_data as data, locations


def get_fitness(self):
    """A simple objective function to calculate distances for the VRP problem.
    Returns:
        int: the total distance of the route
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


P = Population(size=20, optim="min", sol_size=len(locations)-1,
                 valid_set=[i for i in range(1, len(locations))], num_vehicles=data['num_vehicles'])
'''
P.evolve(gens=100, xo_prob=0.9, mut_prob=0.15, select=tournament_sel,#
         xo=cycle_xo, mutate=swap_mutation, elitism=True)'''

#hill_climb(pop)
#sim_annealing(pop)

'''

print()
a= [[] for _ in range(data['num_vehicles'])]
fitness = 0
# Loop over each vehicle's route
for route in [[] for _ in range(data['num_vehicles'])]:
    print(data['distance_matrix'][data['depot']])
    print([route])
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
'''


for individual in P.individuals:
    for route in individual.representation:
        print(route)
    print('-' * 20)


test_individual = P.individuals[0]
fitness = 0
for route in test_individual.representation:
    print(f"Route: {route}")
    if not route:
        print('b')
        continue
    # Add the distance from the depot to the first location
    fitness += data['distance_matrix'][data['depot']][route[0]]
    print(f"{data['distance_matrix'][data['depot']][route[0]]}, Depot to First Location")
    # Add the distances between consecutive locations in the route
    for i in range(len(route) - 1):
        fitness += data['distance_matrix'][route[i]][route[i + 1]]
        print(f"{data['distance_matrix'][route[i]][route[i + 1]]}, Between Locations {route[i]} and {route[i + 1]}")
    # Add the distance from the last location back to the depot
    fitness += data['distance_matrix'][route[-1]][data['depot']]
    print(f"{data['distance_matrix'][route[-1]][data['depot']]}, Last Location to Depot")

print(f"Total Fitness: {fitness}")
