from operator import attrgetter
from random import shuffle, choice, sample, random
from copy import copy
import numpy as np
import sys

# Import TSP data
sys.path.insert(0, 'data')  # Adjust this if your data.py is located elsewhere
from data import data

def vrp_fitness(individual):
    total_distance = 0
    # Loop over each vehicle's route
    for route in individual.representation:
        if not route:  
            continue
        # Add the distance from the depot to the first location
        total_distance += data['distance_matrix'][data['depot']][route[0]]
        # Add the distances between consecutive locations in the route
        for i in range(len(route) - 1):
            total_distance += data['distance_matrix'][route[i]][route[i + 1]]
        # Add the distance from the last location back to the depot
        total_distance += data['distance_matrix'][route[-1]][data['depot']]
        
    return total_distance

# TSP-specific crossover
def crossover(parent1, parent2):
    size = len(parent1)
    cxpoint1, cxpoint2 = sorted(sample(range(size), 2))
    offspring1 = [None] * size
    offspring2 = [None] * size
    offspring1[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    offspring2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2]

    fill = lambda offspring, parent: [item for item in parent if item not in offspring]

    offspring1[:cxpoint1] = fill(offspring1, parent2)
    offspring1[cxpoint2:] = fill(offspring1, parent2)

    offspring2[:cxpoint1] = fill(offspring2, parent1)
    offspring2[cxpoint2:] = fill(offspring2, parent1)

    return offspring1, offspring2

# TSP-specific mutation (swap mutation)
def mutate(individual):
    size = len(individual)
    idx1, idx2 = sample(range(size), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# TSP-specific selection (tournament selection)
def select(population):
    tournament = sample(population.individuals, k=3)
    if population.optim == "max":
        return max(tournament, key=attrgetter('fitness'))
    else:
        return min(tournament, key=attrgetter('fitness'))



class Individual:
    # we always initialize
    def __init__(self, representation=None, size=None, valid_set=None):

        #Removed repetition and creates a representation of the route for each vehicle
        if representation is None:
            self.representation = [[] for _ in range(data['num_vehicles'])]
            locations = sample(valid_set, size)
            for i, location in enumerate(locations):
                self.representation[i % data['num_vehicles']].append(location)
        else:
            self.representation = representation

        # fitness will be assigned to the individual
        self.fitness = self.get_fitness()

    # methods for the class
    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness function.")

    def get_neighbours(self):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f" Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, individuals=[], **kwargs):
        self.size = size
        self.optim = optim
        self.individuals = []
        self.elite= []

        # appending the population with individuals
        for _ in range(size):
            self.individuals.append(
                Individual(
                    size=kwargs["sol_size"],
                    valid_set=kwargs["valid_set"],
                )
            )
    def evolve(self, xo_prob, mut_prob, select, xo, mutate, start, end, elitism=True, gens=100):
        last_return = []
        best_fitness = None
        last_improved_gen = start
        
        for i in range(gens):
            new_pop = []

            if elitism:
                if self.optim == "max":
                    elite = copy(max(self.individuals, key=attrgetter('fitness')))
                elif self.optim == "min":
                    elite = copy(min(self.individuals, key=attrgetter('fitness')))

                new_pop.append(elite)

            while len(new_pop) < self.size:
                # selection
                parent1, parent2 = select(self), select(self)
                # xo with prob
                if random() < xo_prob:
                    offspring1, offspring2 = xo(parent1, parent2)
                # replication
                else:
                    offspring1, offspring2 = parent1, parent2
                # mutation with prob
                if random() < mut_prob:
                    offspring1 = mutate(offspring1)
                if random() < mut_prob:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))
            
            if elitism:
                if self.optim == "max":
                    worst = min(new_pop, key=attrgetter('fitness'))
                    if elite.fitness > worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)
                if self.optim == "min":
                    worst = max(new_pop, key=attrgetter('fitness'))
                    if elite.fitness < worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)

            self.individuals = new_pop

            if self.optim == "max":
                print(f"Best individual of gen #{i + 1}: {max(self, key=attrgetter('fitness'))}")
            elif self.optim == "min":
                print(f"Best individual of gen #{i + 1}: {min(self, key=attrgetter('fitness'))}")


    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]
    
    def __repr__(self):
        return f"Population(size={self.size}, individuals={self.individuals})"


# Assign the fitness function
Individual.get_fitness = vrp_fitness
print("Fitness function assigned successfully.")

# Check data contents
print("Distance Matrix:")
for row in data['distance_matrix']:
    print(row)
print("Number of vehicles:", data['num_vehicles'])
print("Depot:", data['depot'])
print(len(data['distance_matrix']))

# Create a population
try:
    population = Population(
        size=10,
        optim="min",
        sol_size=len(data['distance_matrix']) - 1,  # Exclude depot
        valid_set=list(range(1, len(data['distance_matrix']))),  # Locations excluding the depot
        repetition=False  # No longer needed
    )
    print("Population created successfully.")
    print("Best individual: ", population[0].representation, "with fitness: ", population[0].fitness)
except Exception as e:
    print(f"Error creating population: {e}")