import numpy as np

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



if __name__ == '__main__':
    p1, p2 = [ [288, 50.388445, 4.8229103, "BOIS-DE-VILLERS"], [291, 50.6009713, 4.9284973, "BOLINNE"], [294, 50.3768155, 5.52394, "BOMAL-SUR-OURTHE"] ], [ [307, 51.0873534, 2.7440942, "BOOITSHOEKE"], [310, 50.9803397, 4.5744373, "BOORTMEERBEEK"], [313, 50.7962111, 5.3462923, "BORGLOON"]]
    o1, o2 = geometric_xo(p1, p2)
    print(o1, o2)
