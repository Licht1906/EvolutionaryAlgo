import numpy as np
import random
import copy
import matplotlib.pyplot as plt

###############################

class Problem:
    def __init__(self):
        pass

###############################

def decode(chromosome, problem: Problem): #decoding for gene encoding
    pass
def get_fitness(x):
    pass

###############################

class Individual:
    def __init__(self):
        self.chromosome = None
        self.fitness = None

    def genIndi(self, problem: Problem):
        self.chromosome

    def cal_fitness(self, problem):
        self.fitness = get_fitness(self.chromosome)

    def clone(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        return f"chromosome={self.chromosome}, fitness={self.fitness}"
    
###############################

def crossover(parent1, parent2, problem: Problem, eta = 2.0):
    off1 = Individual()
    off2 = Individual()
    r = np.random.rand()
    if (r <= 0.5):
        beta = (2 * r) ** (1.0/(eta + 1))
    else:
        beta = (1.0/(2*(1 - r))) ** (1.0/(eta + 1))
    
    p1 = parent1.chromosome
    p2 = parent2.chromosome

    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 + beta) * p2 + (1 - beta) * p1)

    c1 = np.clip(c1, 0.0, 1.0)
    c2 = np.clip(c2, 0.0, 1.0)

    off1.chromosome = c1
    off2.chromosome = c2

    return off1.clone(), off2.clone()


###############################

def mutation(ind, eta = 2.0):
    chr = ind.chromosome
    for i in range(chr.size):
        mu = np.random.rand()
        if (mu <= 0.5):
            delta = (2 * mu) ** (1.0/(1 + eta)) - 1
            chr[i] = chr[i] + delta * chr[i]
        else:
            delta = 1 - (2 - 2 * mu) ** (1.0/(1 + eta))
            chr[i] = chr[i] + delta * (1 - chr[i])
        
    chr = np.clip(chr, 0.0, 1.0)
    ind.chromosome = chr
    return ind.clone()

###############################

class Population:
    def __init__(self, pop_size, problem: Problem):
        self.pop_size = pop_size
        self.list_ind = []
        self.problem = problem

    def genPop(self):
        for i in range(self.pop_size):
            ind = Individual()
            ind.genIndi(self.problem)
            ind.cal_fitness(self.problem)
            self.list_ind.append(ind)

    def __repr__(self):
        pass 

###############################

def selection(list, k = 2):
    tour1 = random.sample(list, k)
    tour2 = random.sample(list, k)
    x = max(tour1, key=lambda ind: ind.fitness)
    y = max(tour2, key=lambda ind: ind.fitness)
    return x.clone(), y.clone()

###############################

def survival_selection(list, pop_size):
    list = sorted(list, key=lambda ind: ind.fitness, reverse=True)
    list = list[0: pop_size]
    return list

###############################

def GA(problem, pop_size, max_gen, pc, pm):
    pop = Population(pop_size, problem)
    pop.genPop()
    history = []
    for i in range(max_gen):
        child = []
        while(len(child) < pop_size):
            p1, p2 = selection(pop.list_ind)
            if (np.random.rand() <= pc):
                c1, c2 = crossover(p1, p2, problem)
                c1.cal_fitness(problem)
                c2.cal_fitness(problem)
                child.append(c1)
                child.append(c2)
            if (np.random.rand() <= pm):
                p1 = mutation(p1)
                p2 = mutation(p2)
                p1.cal_fitness(problem)
                p2.cal_fitness(problem)
                child.append(p1)
                child.append(p2)
        pop.list_ind = survival_selection(pop.list_ind + child, pop_size)
        history.append(pop.list_ind[0].fitness)
    solution = pop.list_ind[0]
    return history, solution

###############################

# setup
problem = Problem()

pop_size = 200
max_gen = 2000
Pc = 0.9
Pm = 0.2

# start
fitness_history, solution = GA(problem, pop_size, max_gen, Pc, Pm)

#show
for i in range(len(fitness_history)):
    print(f"Generation {i}, bestfitness = {fitness_history[i]:.2f}")

#show
np.set_printoptions(precision=2, suppress=True)
print("solution:")
print(decode(solution.chromosome, problem))
print(f"{solution.fitness:.2f}")

generations = list(range(len(fitness_history)))
plt.figure(figsize=(10, 5))
plt.plot(generations, fitness_history, marker='o', linestyle='-', color='b', label='Best Fitness')

plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness Progress Over Generations")
plt.legend()
plt.grid(True)
plt.show()