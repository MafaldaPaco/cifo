import numpy as np
from random import sample

def geometric_xo(p1, p2):
    offspring1 = [[None for elem in layer] for layer in p1]  # Initialize offspring with the structure of parent 1
    offspring2 = [[None for elem in layer] for layer in p2]  # Initialize offspring with the structure of parent 2
    
    for layer in range(len(p1)):
        for elem in range(len(p1[layer])):
            if elem % 3 == 1:  # Coordinates are in the second column of each layer
                r = np.random.uniform()
                offspring1[layer][elem] = r * p1[layer][elem] + (1 - r) * p2[layer][elem]
                offspring2[layer][elem] = r * p2[layer][elem] + (1 - r) * p1[layer][elem]
            else: 
                offspring1[layer][elem] = p1[layer][elem]
                offspring2[layer][elem] = p2[layer][elem]

    return offspring1, offspring2

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


if __name__ == '__main__':
    p1, p2 = [ [288, 50.388445, 4.8229103, "BOIS-DE-VILLERS"], [291, 50.6009713, 4.9284973, "BOLINNE"], [294, 50.3768155, 5.52394, "BOMAL-SUR-OURTHE"] ], [ [307, 51.0873534, 2.7440942, "BOOITSHOEKE"], [310, 50.9803397, 4.5744373, "BOORTMEERBEEK"], [313, 50.7962111, 5.3462923, "BORGLOON"]]
    o1, o2 = geometric_xo(p1, p2)
    print(o1, o2)
