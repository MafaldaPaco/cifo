from charles.charles import Population, Individual
from charles.selection import fps, tournament_sel, sigma_scaling_selection, rank_selection
from charles.mutation import swap_mutation, inversion_mutation, binary_mutation, shuffle_mutation
from charles.crossover import order_crossover, single_point_xo, position_based_crossover, geometric_xo
from data.data import vrp_data as data, locations
from operator import attrgetter
import csv
import os

#Redefining the get_fitness from vrp
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

# Monkey patching
Individual.get_fitness = get_fitness


# Auxiliary logger function
def list2csv(mylist, csvfile, header):
    with open(csvfile, 'a', newline='') as f:
        fcsv = csv.writer(f)
        needs_header = os.stat(csvfile).st_size == 0
        if needs_header:
            fcsv.writerow(header)
        for row in mylist:
            fcsv.writerow(row)

#Different combinations of genetic operators and parameters
generations = [100, 200]
pop_size = [100, 150]
elite_size = [0, 1, 3]
xo_probs = [0.99, 0.9]
mut_probs = [0.2, 0.1]
plateau_tolerance = [20, 500]

crossover_functions = [order_crossover, single_point_xo, position_based_crossover, geometric_xo]
selection_functions = [tournament_sel, sigma_scaling_selection, rank_selection]
mutation_functions = [swap_mutation, inversion_mutation, shuffle_mutation]


#Nested for loops to test these combinations
for xo in crossover_functions:
    for sel in selection_functions:
        for mut in mutation_functions:
            for p_xo in xo_probs:
                for p_mut in mut_probs:
                    for pop in pop_size:
                        for gen in generations:
                            for elite in elite_size:
                                for tolerance in plateau_tolerance:
                                    for run in range(1, 5): # perform 5 runs
                                                                             
                                        # Perform the evolution process with current combination of functions
                                        P = Population(size=pop, optim="min", sol_size=len(locations)-1, valid_set=[i for i in range(1, len(locations))], num_vehicles=data['num_vehicles'])
                                        
                                        log = []

                                        results= P.evolve(gens=gen, xo_prob=p_xo, mut_prob=p_mut, select=sel, xo=xo, mutate=mut, elitism=elite, plateau_tolerance= tolerance)
                                    
                                        #Logging results 
                                        for result in results:
                                            log.append(result)
                         
                                        #Output results 
                                        print("Fitness: {}. {} generations, population size {}, xo probability {}, mut probability {}, elit: {}, Sel={}, XO={}, Mut={}".format( min(P, key=attrgetter("fitness")).fitness, generations, pop, p_xo, p_mut, elite, sel.__name__, xo.__name__, mut.__name__))

                                        list2csv(log, "recent.csv", header=["Fitness", "Best fitness", "Generation", "Last improved geneneration", "Xo prob", "Mut prob", "Selection", "Crossover", "Mutation", "Generations", "Elitism", "Tolerance"])
