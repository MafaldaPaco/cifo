from charles.charles import Population, Individual
from charles.selection import fps, tournament_sel, sigma_scaling_selection, rank_selection
from charles.mutation import swap_mutation, inversion_mutation, binary_mutation, shuffle_mutation
from charles.crossover import order_crossover, single_point_xo, position_based_crossover, geometric_xo
from data.data import vrp_data as data, locations
import time
from operator import attrgetter
import uuid
import csv
import os

def get_fitness(self):
    """A simple objective function to calculate distances for the VRP problem, with a penalization for excessive use of vehicles.
    Returns:
        int: the total distance of the route
    """
    fitness = 0
    vehicles_used = sum(1 for route in self.representation if route)
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
    
    # Penalizing for use of more vehicles than necessary
    penalty = 0
    if vehicles_used > 1:
        penalty = (vehicles_used-1) * 100
    
    return fitness + penalty

# Monkey patching
Individual.get_fitness = get_fitness



# Used for logging
def dict2csv(dict, csvfile):
    if csvfile:
        f = open(csvfile, 'a', newline='')
        fcsv = csv.writer(f)
        needs_header = os.stat(csvfile).st_size == 0
        if needs_header:
            fcsv.writerow(list(dict))
            needs_header = False
        # Then, write values
        fcsv.writerow(dict.values())

        f.close()


# Used for logging
def list2csv(mylist, csvfile, header):
    if csvfile:
        f = open(csvfile, 'a', newline='')
        fcsv = csv.writer(f)
        needs_header = os.stat(csvfile).st_size == 0
        if needs_header:
            fcsv.writerow(header)
            needs_header = False
        # Then, write values
        fcsv.writerow(mylist)
        f.close()

# Nested for loops to test different combinations of genetic operators and parameters

generations = [100, 150]
pop_size = [100, 150]
elite_size = [1, 3]
xo_probs = [0.99]
mut_probs = [0.2]
plateau_tolerance = [10, 20, 500]

crossover_functions = [order_crossover, single_point_xo, position_based_crossover, geometric_xo]
selection_functions = [tournament_sel, sigma_scaling_selection, rank_selection]
mutation_functions = [swap_mutation, inversion_mutation, binary_mutation, shuffle_mutation]

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

                                        rundict = {}
                                        rundict["runkey"]                   = uuid.uuid4()
                                        rundict["popsize"]                  = pop
                                        rundict["generations"]              = gen
                                        rundict["xo_prob"]                  = p_xo
                                        rundict["mut_prob"]                 = p_mut
                                        rundict["selection_function"]       = sel
                                        rundict["mutation_function"]        = mut
                                        rundict["xo_function"]              = xo
                                        rundict["elitism"]                  = elite
                                        rundict["patience"]                 = tolerance
                                        rundict["runcount"]                 = run
                                        
                                        # Perform the evolution process with current combination of functions
                                        P = Population(size=pop, optim="min", sol_size=len(locations)-1, valid_set=[i for i in range(1, len(locations))], num_vehicles=data['num_vehicles'])
                                        
                                        start = time.time()
                                        detaillog = []

                                        results= P.evolve(gens=gen, xo_prob=p_xo, mut_prob=p_mut, select=sel, xo=xo, mutate=mut, elitism=elite, plateau_tolerance= tolerance)
                                        
                                        end = time.time()
                                        
                                        # code used for logging results do CSV
                                        for xitem in results:
                                            detaillog.append(xitem)
                                        
                                        print(f'Best individual @gen {generations}: {min(P, key=attrgetter("fitness"))}')
                                        
                                        # store execution parameters in the dictionary, so that they're saved to the logfile.
                                        rundict["bestfit"] = min(P, key=attrgetter("fitness")).fitness
                                        rundict["duration_secs"] = end - start
                                        rundict["selection_function"] = rundict["selection_function"].__name__
                                        rundict["xo_function"] = rundict["xo_function"].__name__
                                        rundict["mutation_function"] = rundict["mutation_function"].__name__


                                        # output results to the console
                                        print( "Fitness (Accuracy): {}. {} generations, pop size {}, xo prob {}, mut prob {}, Elit: {}, Sel={}, XO={}, Mut={}. Took {:.2f} secs.".format( min(P, key=attrgetter("fitness")).fitness, rundict["generations"], rundict["popsize"], rundict["xo_prob"], rundict["mut_prob"], rundict["elitism"], rundict["selection_function"], rundict["xo_function"], rundict["mutation_function"], end - start))

                                        # Record logs of the whole run
                                        dict2csv(rundict, "sunday_log.csv")

                                        # Record detailed logs, with results after each generation
                                        list2csv(detaillog, "sunday_detail.csv", header=["Fitness", "Generation", "Last improved gen"])


