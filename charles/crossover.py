from random import randint, sample, uniform
import numpy as np
from charles.charles import Population, Individual

def single_point_xo(parent1, parent2):
    """Implementation of single point crossover.

    Args:
        parent1 (Individual): First parent for crossover.
        parent2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    xo_point = randint(1, len(parent1)-1)
    offspring1 = parent1[:xo_point] + parent2[xo_point:]
    offspring2 = parent2[:xo_point] + parent1[xo_point:]
    return offspring1, offspring2


'''def cycle_xo(p1, p2):
    """Implementation of cycle crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    # offspring placeholders
    sublist_lengths = [len(sublist) for sublist in p1.representation]

    offspring1 = [[None] * length for length in sublist_lengths]
    offspring2 = [[None] * length for length in sublist_lengths]
    while None in offspring1:
        index = offspring1.index(None)
        val1 = p1[index]
        val2 = p2[index]

        # copy the cycle elements
        while val1 != val2:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]
            val2 = p2[index]
            index = p1.index(val2)

        # copy the rest
        for element in offspring1:
            if element is None:
                index = offspring1.index(None)
                if offspring1[index] is None:
                    offspring1[index] = p2[index]
                    offspring2[index] = p1[index]

    return offspring1, offspring2


def pmx(p1, p2):
    """Implementation of partially matched/mapped crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    xo_points = sample(range(len(p1)), 2)
    #xo_points = [3,6]
    xo_points.sort()

    def pmx_offspring(x,y):
        o = [None] * len(x)
        # offspring2
        o[xo_points[0]:xo_points[1]]  = x[xo_points[0]:xo_points[1]]
        z = set(y[xo_points[0]:xo_points[1]]) - set(x[xo_points[0]:xo_points[1]])

        # numbers that exist in the segment
        for i in z:
            temp = i
            index = y.index(x[y.index(temp)])
            while o[index] is not None:
                temp = index
                index = y.index(x[temp])
            o[index] = i

        # numbers that doesn't exist in the segment
        while None in o:
            index = o.index(None)
            o[index] = y[index]
        return o

    o1, o2 = pmx_offspring(p1, p2), pmx_offspring(p2, p1)
    return o1, o2


def geo_xo(p1,p2):
    """Implementation of arithmetic crossover/geometric crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individual: Offspring, resulting from the crossover.
    """
    o = [None] * len(p1)
    for i in range(len(p1)):
        r = uniform(0,1)
        o[i] = p1[i] * r + (1-r) * p2[i]
    return o'''


# Crossover functions
def order_crossover(p1, p2):
    size = len(p1)
    cx1, cx2 = np.sort(np.random.choice(size + 1, 2, replace=False))

    missing1 = [gene for gene in p2 if gene not in p1[cx1:cx2]]
    missing2 = [gene for gene in p1 if gene not in p2[cx1:cx2]]

    offspring1, offspring2 = [[] for _ in range(size)], [[] for _ in range(size)]

    offspring1[:cx1] = missing1[:cx1]
    offspring2[:cx1] = missing2[:cx1]
    offspring1[cx1:cx2] = p1[cx1:cx2]
    offspring2[cx1:cx2] = p2[cx1:cx2]
    offspring1[cx2:] = missing1[cx1:]
    offspring2[cx2:] = missing2[cx1:]

    return offspring1, offspring2


'''def order_based_crossover(parent1, parent2):
    size = len(parent1)
    selected = np.random.choice([True, False], size)

    matching1 = np.isin(parent2, parent1[selected])
    matching2 = np.isin(parent1, parent2[selected])

    offspring1, offspring2 = np.empty(size, dtype=int), np.empty(size, dtype=int)

    offspring1[matching1] = parent1[selected]
    offspring2[matching2] = parent2[selected]
    offspring1[~matching1] = parent2[~matching1]
    offspring2[~matching2] = parent1[~matching2]

    return offspring1, offspring2'''

def position_based_crossover(p1, p2):
    size = len(p1)
    selected = np.random.choice([True, False], size)

    sublist_lengths = [len(sublist) for sublist in p1.representation]

    offspring1, offspring2 = [[] *length for length in sublist_lengths], [[None] *length  for length in sublist_lengths]

    # Select elements based on 'selected'
    selected_indices = np.where(selected)[0]
    unselected_indices = np.where(~selected)[0]
    for e in selected_indices:
        offspring1[e] = p1[e]
        offspring2[e] = p2[e]

    # Reverse the selection process
    for index in unselected_indices:
        if p2[index] not in offspring1:
            offspring1[index] = p2[index]
        if p1[index] not in offspring2:
            offspring2[index] = p1[index]

    return offspring1, offspring2

'''def cycle_crossover(parent1, parent2):
    size = len(parent1)
    offspring1, offspring2 = np.empty(size, dtype=int), np.empty(size, dtype=int)
    visited = np.zeros(size, dtype=bool)
    source1, source2 = parent1, parent2

    for start in range(size):
        if visited[start]:
            continue
        visited[start] = True
        cycle = [start]

        while parent2[cycle[-1]] != parent1[start]:
            next_index = np.where(parent1 == parent2[cycle[-1]])[0][0]
            cycle.append(next_index)
            visited[next_index] = True

        cycle = np.array(cycle)
        offspring1[cycle] = source1[cycle]
        offspring2[cycle] = source2[cycle]

        source1, source2 = source2, source1

    return offspring1, offspring2

def partially_mapped_crossover(parent1, parent2):
    size = len(parent1)
    cx1, cx2 = np.sort(np.random.choice(size + 1, 2, replace=False))

    def one_offspring(p1, p2):
        offspring = np.empty(size, dtype=int)
        offspring[cx1:cx2] = p1[cx1:cx2]

        for i in np.concatenate([np.arange(0, cx1), np.arange(cx2, size)]):
            candidate = p2[i]
            while candidate in p1[cx1:cx2]:
                candidate = p2[np.where(p1 == candidate)[0][0]]
            offspring[i] = candidate
        return offspring

    return one_offspring(parent1, parent2), one_offspring(parent2, parent1)

def edge_recombination_crossover(parent1, parent2):
    size = len(parent1)
    edges = {}

    for i, v in enumerate(parent1):
        j = np.where(parent2 == v)[0][0]
        edges[v] = [parent1[(i - 1) % size], parent1[(i + 1) % size],
                    parent2[(j - 1) % size], parent2[(j + 1) % size]]

    def one_offspring(parent):
        offspring = np.empty(size, dtype=int)
        missing = list(parent[1:])
        offspring[0] = parent[0]
        node = parent[0]

        for index in range(1, size):
            nodes = np.array(edges[node])
            nodes = nodes[np.in1d(nodes, missing)]
            next_node = np.random.choice(nodes) if nodes.size else np.random.choice(missing)
            offspring[index] = next_node
            node = next_node
            missing.remove(next_node)

        return offspring

    return one_offspring(parent1), one_offspring(parent2)'''

def geometric_xo(p1, p2):
    size = len(p1)
    child1, child2 = [-1]*size, [-1]*size

    for i in range(size):
        if np.random.rand() > 0.5:
            child1[i], child2[i] = p1[i], p2[i]
        else:
            child1[i], child2[i] = p2[i], p1[i]

    return child1, child2

if __name__ == "__main__":
    #p1, p2 = [9,8,2,1,7,4,5,10,6,3], [1,2,3,4,5,6,7,8,9,10]
    #p1, p2 = [2,7,4,3,1,5,6,9,8], [1,2,3,4,5,6,7,8,9]
    p1, p2 = [9,8,4,5,6,7,1,3,2,10], [8,7,1,2,3,10,9,5,4,6]
    o1, o2 = pmx(p1, p2)
    print(o1,o2)

