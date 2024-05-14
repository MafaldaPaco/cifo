import seaborn as sns
import matplotlib.pyplot as plt
from copy import copy
'''
def fitness(number):
    return number**2
    #return "{0:04b}".format(number).count("1")

# plotting the fitness landscape of the int_bin problem
sns.lineplot(y=[fitness(i) for i in range(1, 16)], x=[i for i in range (1,16)])
plt.show()



sol = [0,0,0,0]

n = [copy(sol) for _ in range(len(sol))]

for i, ne in enumerate(n):
    if ne[i] == 1:
        ne[i] = 0
    elif ne[i] == 0:
        ne[i] = 1

print(n)

'''

# simulated annealing parameters

def plot_c(c, alpha, threshold):
    c_list = [c]
    while c > threshold:
        c = c * alpha
        c_list.append(c)
    plt.plot(c_list)
    plt.show()
    print(c_list)

plot_c(10,0.95,0.05)













