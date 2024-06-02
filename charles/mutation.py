from random import randint, sample
import numpy as np

# Binary mutation changes a single bit in the individual's representation.
# This is useful for binary encoded individuals, where a bit flip can introduce small changes.
def binary_mutation(individual):
    """Binary mutation for a GA individual

    Args:
        individual (Individual): A GA individual from charles.py

    Raises:
        Exception: When individual is not binary encoded.py

    Returns:
        Individual: Mutated Individual
    """
    # Randomly select a mutation index
    mut_index = randint(0, len(individual)-1)
    # Flip the bit at the mutation index
    if individual[mut_index] == 1:
        individual[mut_index] = 0
    elif individual[mut_index] == 0:
        individual[mut_index] = 1
    else:
        raise Exception("Representation is not binary!")

    return individual

# Swap mutation exchanges the positions of two genes in the individual's representation.
# This can introduce larger changes than binary mutation, which is useful for permutation-based representations.
def swap_mutation(individual):
    """Swap mutation for a GA individual. Swaps the bits.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    # Randomly select two positions to swap
    mut_indexes = sample(range(0, len(individual)), 2)
    # Swap the selected positions
    individual[mut_indexes[0]], individual[mut_indexes[1]] = individual[mut_indexes[1]], individual[mut_indexes[0]]
    return individual

# Inversion mutation reverses the order of a subset of genes in the individual's representation.
# This can significantly alter the sequence, which is useful for route optimization problems.
def inversion_mutation(individual):
    """Inversion mutation for a GA individual. Reverts a portion of the representation.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    # Randomly select two positions to define the subset
    mut_indexes = sample(range(0, len(individual)), 2)
    mut_indexes.sort()
    # Reverse the order of the genes in the subset
    individual[mut_indexes[0]:mut_indexes[1]] = individual[mut_indexes[0]:mut_indexes[1]][::-1]
    return individual

# Shuffle mutation randomly shuffles a subset of genes in the individual's representation.
# This can introduce significant variation while preserving the overall gene set.
def shuffle_mutation(individual):
    size = len(individual)
    # Randomly select positions to shuffle
    indices = np.random.choice([True, False], size)
    indices = np.where(indices)[0]
    subset = [individual[i] for i in indices]
    # Shuffle the selected positions
    np.random.shuffle(subset)
    for i, e in enumerate(indices):
        individual[e] = subset[i]
    return individual

# Add/Subtract mutation decreases one gene value and increases another.
# This maintains the total sum of the gene values, useful for certain optimization problems.
def add_subtract_mutation(individual):
    # Randomly select an index with a non-zero value
    idx1 = np.random.choice(np.nonzero(individual)[0])
    idx2 = idx1
    while idx2 == idx1:
        idx2 = np.random.randint(len(individual))

    # Subtract from one position and add to another
    individual[idx1] -= 1
    individual[idx2] += 1
    return individual

# Swap mutation exchanges the positions of two genes with a certain probability.
# This version includes a mutation rate parameter to control the likelihood of mutation.
def swap_mutation(individual, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.random.choice(len(individual), 2, replace=False)
        # Swap the selected positions
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Inversion mutation reverses the order of a subset of genes with a certain probability.
# This version includes a mutation rate parameter to control the likelihood of mutation.
def inversion_mutation(individual, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        idx1, idx2 = np.sort(np.random.choice(len(individual), 2, replace=False))
        # Reverse the order of the genes in the subset
        individual[idx1:idx2+1] = individual[idx1:idx2+1][::-1]
    return individual

if __name__ == "__main__":
    test = [3, 1, 2, 4, 5, 6, 3]
    inversion_mutation(test)
