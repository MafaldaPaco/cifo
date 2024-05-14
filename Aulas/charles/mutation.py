from random import randint, sample


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
    mut_indexes = sample(range(0, len(individual)), 2)
    individual[mut_indexes[0]], individual[mut_indexes[1]] = individual[mut_indexes[1]], individual[mut_indexes[0]]
    return individual


test = [3,1,2,4]
swap_mutation(test)