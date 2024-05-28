from operator import attrgetter
from random import sample

def select(population):
    tournament = sample(population.individuals, k=3)
    if population.optim == "max":
        return max(tournament, key=attrgetter('fitness'))
    else:
        return min(tournament, key=attrgetter('fitness'))