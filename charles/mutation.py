from random import randint, sample
import numpy as np

def binary_mutation(individual):
    """Binary mutation for a GA individual

    Args:
        individual (Individual): A GA individual from charles.py

    Raises:
        Exception: When individual is not binary encoded.py

    Returns:
        Individual: Mutated Individual
    """
    mut_index = randint(0, len(individual)-1)
    if individual[mut_index] == 1:
        individual[mut_index] = 0
    elif individual[mut_index] == 0:
        individual[mut_index] = 1
    else:
        raise Exception("Representation is not binary!")

    return individual


def swap_mutation(individual):
    """Swap mutation for a GA individual. Swaps the bits.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_indexes = sample(range(0, len(individual)), 2)
    individual[mut_indexes[0]], individual[mut_indexes[1]] = individual[mut_indexes[1]], individual[mut_indexes[0]]
    return individual


def inversion_mutation(individual):
    """Inversion mutation for a GA individual. Reverts a portion of the representation.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_indexes = sample(range(0, len(individual)), 2)
    mut_indexes.sort()
    individual[mut_indexes[0]:mut_indexes[1]] = individual[mut_indexes[0]:mut_indexes[1]][::-1]
    return individual

# Mutation functions
def shuffle_mutation(individual):
    size = len(individual)
    indices = np.random.choice([True, False], size)
    subset = individual[indices]
    np.random.shuffle(subset)
    individual[indices] = subset

def add_subtract_mutation(individual):
    idx1 = np.random.choice(np.nonzero(individual)[0])
    idx2 = idx1
    while idx2 == idx1:
        idx2 = np.random.randint(len(individual))

    individual[idx1] -= 1
    individual[idx2] += 1

def swap_mutation(individual, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

def inversion_mutation(individual, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.sort(np.random.choice(len(individual), 2, replace=False))
        individual[idx1:idx2+1] = individual[idx1:idx2+1][::-1]

def inversion_mutation(individual, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.sort(np.random.choice(len(individual), 2, replace=False))
        individual[idx1:idx2+1] = individual[idx1:idx2+1][::-1]



if __name__ == "__main__":
    test = [3,1,2,4,5,6,3]
    inversion_mutation(test)