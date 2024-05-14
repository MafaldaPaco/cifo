from charles.charles import Population, Individual
from charles.search import hill_climb
from copy import copy


def get_fitness(self):
    """A fitness function that returns the
    number of 1's occurring in the binary representation.

    Alternatively, the square of the number.

    Returns:
        int: the number of 1's in the binary representation.

    Alternatively int: the square of the number.
    """
    #return "{0:04b}".format(self.representation[0]).count("1")
    #return (self.representation[0])**2
    return self.representation.count(1)


def get_neighbours(self):
    """A neighbourhood function for the int_bin problem.
    +-1 of the integer.
    Alternatively, flips the bits

    Returns:
        list: a list of individuals
    """
    n = [copy(self.representation) for _ in range(len(self.representation))]

    for i, ne in enumerate(n):
        if ne[i] == 1:
            ne[i] = 0
        elif ne[i] == 0:
            ne[i] = 1

    n = [Individual(ne) for ne in n]
    return n


'''
    if self.representation[0] == 1:
        return (Individual(representation=[self.representation[0] + 1]),)

    if self.representation[0] == 15:
        return (Individual(representation=[self.representation[0] - 1]),)

    n1 = Individual(representation=[self.representation[0] + 1])
    n2 = Individual(representation=[self.representation[0] - 1])
    return n1, n2
'''

# Monkey patching
Individual.get_fitness = get_fitness
Individual.get_neighbours = get_neighbours

pop = Population(size=1, optim="max", sol_size=4, valid_set=[0, 1])
#pop = Population(size=1, optim="max", sol_size=1, valid_set=[i for i in range(1, 16)])


hill_climb(pop)
