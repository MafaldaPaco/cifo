from random import sample

def mutate(individual):
    size = len(individual)
    idx1, idx2 = sample(range(size), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

