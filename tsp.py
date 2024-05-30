#Genetic algorithm to solve Travelling salesmann problem
#Author : Syed Anjum Haider Rizvi

import os
import numpy as np
import matplotlib.pyplot as plt

town = np.array([[10,10],
                 [7,5],
                 [50,0],
                 [20,7],
                 [40,35],
                 [60,95],
                 [72,85],
                 [70,10],
                 [30,90],
                 [20,30]])

def compute_fitness(individual):
    lst = []
    res = 0.0
    for i in individual:
        lst.append(i)
    for j in range(len(individual)-1):
        res += np.linalg.norm(town[lst[j],:]-town[lst[j+1],:])
    return -res

def individual_p(maps):
    #Individual element of the population
    return np.random.choice(range(maps), maps, replace=False)

individual = individual_p(len(town))
results = compute_fitness(individual)

#print("fitness",results)
#print("individual==",individual)

def initial_population(n_town,population_sample):
    population = [individual_p(n_town) for _ in range(population_sample)]
    population.sort(key=lambda x: compute_fitness(x))
    return population


def parents(populations,num_childs):
    mothers = population[-2*num_childs::2]
    fathers = population[-2*num_childs+1::2]
    return mothers,fathers

def cross_over(mother,father):
    moth_h = mother[:int(len(mother)*0.5)].copy()
    moth_t = mother[int(len(mother)*0.5):].copy()
    fath_t = father[int(len(father)*0.5):].copy()
    mapping = {fath_t[i]:moth_t[i] for i in range(len(moth_t))}
    for i in range(len(moth_h)):
        while moth_h[i] in fath_t:
            moth_h[i] = mapping[moth_h[i]]
    
    return np.hstack([moth_h,fath_t])

def mutation(child):
    #interchange two random town in the child
    i,j = np.random.choice(range(len(child)),2,replace=False)
    child[i],child[j] = child[j],child[i]
    return child

def update_population(population,new_childs):
    population.extend(new_childs)
    population.sort(key=lambda x:compute_fitness(x))
    return population[-len(population):]


generations=200
population_size = 100
num_childs = 30

fitness_each_gen = []
population = initial_population(len(town),population_size)

for i in range(generations):
    mothers,fathers = parents(population,num_childs)
    childs = []
    for mother,father in zip(mothers,fathers):
        child = mutation(cross_over(mother,father))
        childs.append(child)
    new_population = update_population(population,childs)
    population = new_population
    optimal_p = population[-1]
    fitness_each_gen.append(compute_fitness(optimal_p))
    print("generation at",i)


plt.plot(fitness_each_gen)
plt.title("change in fitness with each generation")
plt.xlabel("generations")
plt.ylabel("fitness")
plt.show()
