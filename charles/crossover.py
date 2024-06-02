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
    # The single point crossover - Randomly select a crossover point
    # This point determines where the two parent genomes will be split and combined, By choosing a random point, 
    # we introduce diversity into the offspring, which can help in exploring new and potentially better solutions.
    # It helps maintain the sequence of bus stops, potentially preserving logical and efficient route structures.
    
    xo_point = randint(1, len(parent1)-1)

    # Create offspring by combining parts of the parents.
    # The offspring will inherit the first part of the first parent up to the crossover point, and the second part of the second parent after the crossover point.
    # This ensures that the offspring inherit genetic material from both parents.

    offspring1 = parent1[:xo_point] + parent2[xo_point:]
    offspring2 = parent2[:xo_point] + parent1[xo_point:]
    return offspring1, offspring2


# Cycle crossover alternates segments between parents to offspring, ensuring no repetition.
# It preserves the relative order by following cycles of matching indices.
# This can help maintain the sequence of bus stops, potentially preserving logical and efficient route structures.

'''def cycle_xo(p1, p2):
    """Implementation of cycle crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """I


    # Initialize placeholders for offspring with the same structure as the parents
    sublist_lengths = [len(sublist) for sublist in p1.representation]

    offspring1 = [[None] * length for length in sublist_lengths]
    offspring2 = [[None] * length for length in sublist_lengths]
    
    # Continue the process until all elements in offspring1 are filled
    while None in offspring1:
    # Start the cycle at the first empty index
        index = offspring1.index(None)
        val1 = p1[index]
        val2 = p2[index]

        # Copy elements to offspring until the cycle is completed
        while val1 != val2:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]
            val2 = p2[index]
            index = p1.index(val2)

        # Copy remaining elements from the other parent to complete the offspring
        for element in offspring1:
            if element is None:
                index = offspring1.index(None)
                if offspring1[index] is None:
                    offspring1[index] = p2[index]
                    offspring2[index] = p1[index]

    return offspring1, offspring2

# PMX maintains relative order by mapping elements between two crossover points.
# It ensures offspring inherit segments from both parents while avoiding duplicates.

def pmx(p1, p2):
    """Implementation of partially matched/mapped crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    
    # Randomly select two crossover points and sort them
    xo_points = sample(range(len(p1)), 2)
    #xo_points = [3,6]
    xo_points.sort()

    def pmx_offspring(x,y):
        o = [None] * len(x)
    # Copy the segment between crossover points from the first parent to the offspring
        # offspring2
        o[xo_points[0]:xo_points[1]]  = x[xo_points[0]:xo_points[1]]
        z = set(y[xo_points[0]:xo_points[1]]) - set(x[xo_points[0]:xo_points[1]])

    # Handle elements in the segment that need mapping to avoid duplicates
        for i in z:
            temp = i
            index = y.index(x[y.index(temp)])
            while o[index] is not None:
                temp = index
                index = y.index(x[temp])
            o[index] = i

        # numbers that doesn't exist in the segment - Fill in the remaining elements from the second parent
        while None in o:
            index = o.index(None)
            o[index] = y[index]
        return o
    # Generate two offspring by applying PMX on both parents
    o1, o2 = pmx_offspring(p1, p2), pmx_offspring(p2, p1)
    return o1, o2


# Geometric crossover combines genes from parents using a weighted average.
# This method introduces variation by blending parental traits, creating diverse offspring.

def geo_xo(p1,p2):
    """Implementation of arithmetic crossover/geometric crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individual: Offspring, resulting from the crossover.
    """
    # Initialize the offspring with the same length as the parents
    o = [None] * len(p1)
    # Generate each gene in the offspring by linearly combining the corresponding genes of the parents
    for i in range(len(p1)):
        r = uniform(0,1)
    # Each gene is a weighted average of the parents' genes, introducing new variations
        o[i] = p1[i] * r + (1-r) * p2[i]
    return o'''

# Order crossover preserves relative order by transferring a segment and filling missing genes.
# This ensures offspring inherit structured sequences from both parents, maintaining route integrity.
# This method is particularly useful for permutation problems, such as the Traveling Salesman Problem (TSP), where the order of elements is crucial.

# Crossover functions

# Define size of the parents
def order_crossover(p1, p2):
    size = len(p1)
# Randomly select two crossover points and sort them
    cx1, cx2 = np.sort(np.random.choice(size + 1, 2, replace=False))

# Identify genes that are missing in the segments
    missing1 = [gene for gene in p2 if gene not in p1[cx1:cx2]]
    missing2 = [gene for gene in p1 if gene not in p2[cx1:cx2]]

# Initialize offspring with empty lists of the same size as parents
    offspring1, offspring2 = [[] for _ in range(size)], [[] for _ in range(size)]

# Fill the segments before the first crossover point with missing genes
    offspring1[:cx1] = missing1[:cx1]
    offspring2[:cx1] = missing2[:cx1]

# Copy the segments between the crossover points directly from the parents
    offspring1[cx1:cx2] = p1[cx1:cx2]
    offspring2[cx1:cx2] = p2[cx1:cx2]

# Fill the segments after the second crossover point with the remaining missing genes
    offspring1[cx2:] = missing1[cx1:]
    offspring2[cx2:] = missing2[cx1:]

    return offspring1, offspring2

# Order-based crossover maintains relative order by selecting elements from parents.
# It ensures offspring inherit selected traits while preserving sequence integrity.
# PBX is useful for problems where the positions of certain elements are critical and must be maintained.

'''def order_based_crossover(parent1, parent2):
# Randomly select which elements to keep from the first parent
    size = len(parent1)
    selected = np.random.choice([True, False], size)

# Determine which elements from the second parent match the selected elements of the first parent
    matching1 = np.isin(parent2, parent1[selected])
    matching2 = np.isin(parent1, parent2[selected])

    # Initialize offspring arrays
    offspring1, offspring2 = np.empty(size, dtype=int), np.empty(size, dtype=int)

    # Assign selected elements to the corresponding positions in offspring
    offspring1[matching1] = parent1[selected]
    offspring2[matching2] = parent2[selected]
 # Fill the remaining positions with elements from the other parent
    offspring1[~matching1] = parent2[~matching1]
    offspring2[~matching2] = parent1[~matching2]

    return offspring1, offspring2'''

# Position-based crossover selects elements based on their positions in the parents.
# It ensures unique inheritance by reversing unselected elements, maintaining diversity and order.

def position_based_crossover(p1, p2):
    size = len(p1)
 # Randomly select which positions to keep from the first parent
    selected = np.random.choice([True, False], size)

    sublist_lengths = [len(sublist) for sublist in p1.representation]

# Initialize offspring with empty lists of the same structure as parents
    offspring1, offspring2 = [[] *length for length in sublist_lengths], [[None] *length  for length in sublist_lengths]

    # Select elements based on 'selected'
    selected_indices = np.where(selected)[0]
    unselected_indices = np.where(~selected)[0]
    for e in selected_indices:
        offspring1[e] = p1[e]
        offspring2[e] = p2[e]

    # Reverse the selection process for unselected elements
    for index in unselected_indices:
        if p2[index] not in offspring1:
            offspring1[index] = p2[index]
        if p1[index] not in offspring2:
            offspring2[index] = p1[index]

    return offspring1, offspring2

# Cycle crossover transfers segments from parents to offspring in cycles,
# Cycle crossover alternates segments between parents to offspring in cycles,
# ensuring unique inheritance and maintaining gene order.

'''def cycle_crossover(parent1, parent2):
    size = len(parent1)
# Initialize offspring arrays
    offspring1, offspring2 = np.empty(size, dtype=int), np.empty(size, dtype=int)
# Track visited indices to avoid redundant cycles
    visited = np.zeros(size, dtype=bool)
# Define initial sources for swapping
    source1, source2 = parent1, parent2

    # Iterate through each index to find cycles
    for start in range(size):
        if visited[start]:
            continue
        visited[start] = True
        cycle = [start]

    # Continue the cycle until it returns to the starting point

        while parent2[cycle[-1]] != parent1[start]:
            next_index = np.where(parent1 == parent2[cycle[-1]])[0][0]
            cycle.append(next_index)
            visited[next_index] = True

        cycle = np.array(cycle)
        offspring1[cycle] = source1[cycle]
        offspring2[cycle] = source2[cycle]

        source1, source2 = source2, source1

    return offspring1, offspring2

PMX ensures offspring inherit segments from both parents while resolving duplicates
# by mapping genes, maintaining a consistent sequence of elements.

def partially_mapped_crossover(parent1, parent2):
    size = len(parent1)
    cx1, cx2 = np.sort(np.random.choice(size + 1, 2, replace=False))

    def one_offspring(p1, p2):
        offspring = np.empty(size, dtype=int)
# Copy the segment from the first parent to the offspring
        offspring[cx1:cx2] = p1[cx1:cx2]

        # Fill the remaining positions with the values from the second parent
        for i in np.concatenate([np.arange(0, cx1), np.arange(cx2, size)]):
            candidate = p2[i]
            while candidate in p1[cx1:cx2]:
                candidate = p2[np.where(p1 == candidate)[0][0]]
            offspring[i] = candidate
        return offspring

    return one_offspring(parent1, parent2), one_offspring(parent2, parent1)

# Edge recombination crossover constructs offspring by following parent edges,
# ensuring adjacency preservation and optimized gene order.
# preserving adjacency information to create more natural and optimized routes.

def edge_recombination_crossover(parent1, parent2):
    # Define size of the parents
    size = len(parent1)
    # Create a dictionary to store edges for each gene
    edges = {}

    # Create the edge list for each gene in parent1 and parent2
    for i, v in enumerate(parent1):
        j = np.where(parent2 == v)[0][0]
        edges[v] = [parent1[(i - 1) % size], parent1[(i + 1) % size],
                    parent2[(j - 1) % size], parent2[(j + 1) % size]]

    def one_offspring(parent):
    # Initialize offspring array
        offspring = np.empty(size, dtype=int)
    # List of genes that are yet to be placed in the offspring
        missing = list(parent[1:])
    # Start with the first gene of the parent
        offspring[0] = parent[0]
        node = parent[0]

     # Create offspring by following edges
        for index in range(1, size):
            nodes = np.array(edges[node])
            nodes = nodes[np.in1d(nodes, missing)]
    # Choose the next node from the list of available edges
            next_node = np.random.choice(nodes) if nodes.size else np.random.choice(missing)
            offspring[index] = next_node
            node = next_node
            missing.remove(next_node)

        return offspring

    # Generate two offspring using the edge recombination crossover method
    return one_offspring(parent1), one_offspring(parent2)'''

# Geometric crossover randomly mixes genes from both parents,
# promoting genetic diversity and blending parental traits.
# ensuring a blend of parental traits and introducing genetic diversity.

def geometric_xo(p1, p2):
 # Define size of the parents
    size = len(p1)
# Initialize offspring with -1
    child1, child2 = [-1]*size, [-1]*size

    # Randomly assign genes from parents to offspring
    for i in range(size):
        if np.random.rand() > 0.5:
            child1[i], child2[i] = p1[i], p2[i]
        else:
            child1[i], child2[i] = p2[i], p1[i]

    return child1, child2

if __name__ == "__main__":
# Test parents
# Example parent lists for testing
    #p1, p2 = [9,8,2,1,7,4,5,10,6,3], [1,2,3,4,5,6,7,8,9,10]
    #p1, p2 = [2,7,4,3,1,5,6,9,8], [1,2,3,4,5,6,7,8,9]
    p1, p2 = [9,8,4,5,6,7,1,3,2,10], [8,7,1,2,3,10,9,5,4,6]
 # Perform PMX crossover

    o1, o2 = pmx(p1, p2)
    print(o1,o2)
