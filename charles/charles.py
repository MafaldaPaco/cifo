from operator import attrgetter
from random import sample, random, randint, choice
from copy import copy
import numpy as np
import sys

class Individual:
    # we always initialize
    def __init__(self, num_vehicles, representation=None, size=None, valid_set=None):

        #Removed repetition + created a route representation for the vehicles
        if representation is None:
            self.representation = [[] for _ in range(num_vehicles)]
            locations = sample(valid_set, size)
            #Randomly choosing how many of the vehicles will be used
            vehicles_used = randint(1, num_vehicles)
            # List of vehicles to use
            vehicles = list(range(vehicles_used))
            for location in locations:
                # Randomly assign each location to one of the vehicles used
                vehicle = choice(vehicles)
                self.representation[vehicle].append(location)
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
        for i, sublist in enumerate(self.representation):
            if sublist == value:
                return i
        raise ValueError(f"{value} is not in list")

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
                    num_vehicles=kwargs["num_vehicles"],
                    size=kwargs["sol_size"],
                    valid_set=kwargs["valid_set"],
                )
            )
    def evolve(self, xo_prob, mut_prob, select, xo, mutate, gens=100, elitism=1, plateau_tolerance=5):
        best_fitness = 1000000
        last_improvement = 1
        evolution = []
        
        for gen in range(gens):
            new_pop = []
            
            #Multiple elites implementation
            if elitism > 0:
                if self.optim == "max":
                    elite = sorted(self.individuals, key=attrgetter("fitness"), reverse=True)[:elitism]
                elif self.optim == "min":
                    elite = sorted(self.individuals, key=attrgetter("fitness"))[:elitism]
                new_pop.extend(elite)

            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self) # selection
                
                if random() < xo_prob: # xo with prob
                    offspring1, offspring2 = xo(parent1, parent2) 
                       
                else: # replication
                    offspring1, offspring2 = parent1, parent2
                    
                # mutation with prob
                if random() < mut_prob:
                    offspring1 = mutate(offspring1)
                                        
                if random() < mut_prob:
                    offspring2 = mutate(offspring2)
                    
                new_pop.append(Individual(representation=offspring1, num_vehicles=sum(1 for route in offspring1 if route)))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2, num_vehicles=sum(1 for route in offspring2 if route)))

            self.individuals = new_pop
            
            #Updating the last improved generation information + storing the best and worst fitness
            if self.optim == "max":
                worst = min(new_pop, key=attrgetter('fitness'))
                best = max(self, key=attrgetter("fitness"))
                current_fitness = best.fitness
                if (current_fitness > best_fitness) or best_fitness == 0:
                    best_fitness = current_fitness
                    last_improvement = gen        
            if self.optim == "min":
                worst = max(new_pop, key=attrgetter('fitness'))
                best = min(self, key=attrgetter("fitness"))
                current_fitness = best.fitness
                if (current_fitness < best_fitness) or best_fitness is None:
                    best_fitness = current_fitness
                    last_improvement = gen
                
            #Removing the individual with the worst fitness if it's fitness is worse than the last elite individual
            if elitism:
                if self.optim == "max" and elite[-1].fitness > worst.fitness:
                    new_pop.pop(new_pop.index(worst))
                if self.optim == "min"and elite[-1].fitness < worst.fitness:
                    new_pop.pop(new_pop.index(worst))


            print(f"Best individual of gen #{gen + 1}: {best}, with a fitness of: {current_fitness}")

            #Improvement check point
            if best_fitness is None or current_fitness < best_fitness:
                best_fitness = current_fitness
                last_improvement = gen
            
            xo_factor= 0.2
            mut_factor= 2
            if gen - last_improvement >= plateau_tolerance:
                xo_prob = xo_prob * xo_factor if xo_prob * xo_factor > 0 else 0
                mut_prob = mut_prob * mut_factor if mut_prob * mut_factor > 0 else 0
                
                print(f"{gen - last_improvement} generations without improvement. New values: xo_prob={xo_prob}, mut_prob={mut_prob}. Current best fitness={best_fitness}")

            evolution.append([min(self, key=attrgetter("fitness")).representation, min(self, key=attrgetter("fitness")).fitness, best_fitness, gen, last_improvement, xo_prob, mut_prob, select.__name__, xo.__name__, mutate.__name__, gens, elitism, plateau_tolerance])

        return evolution
            

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]
    
    def __repr__(self):
        return f"Population(size={self.size}, individuals={self.individuals})"