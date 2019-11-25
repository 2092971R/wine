# -*- coding: utf-8 -*-
"""

An implimentation of a genetic algorithm to determine the best attributes to use

@author: Alexandra Russell (201882769)
"""

"""
Setup
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
#size of the population    
popsize = 60
#iterations
iterate = 20


wine = pd.read_csv("winequality-red.csv")
min_max_scaler = preprocessing.MinMaxScaler()
wine_scaled = min_max_scaler.fit_transform(wine)
wine = pd.DataFrame(wine_scaled)
mapping_2={0: "fixed acididty (g/dm^3)", 
                              1: "volatile acididty (g/dm^3)",
                              2: "citric acid (g/dm^3)",
                              3: "residual sugar (g/dm^3)",
                              4: "chlorides (g/dm^3)",
                              5: "free sulfur dioxide (mg/dm^3)",
                              6: "total sulfur dioxide (mg/dm^3)",
                              7: "density(g/cm^3)",
                              8: "pH",
                              9: "sulphates (g/dm^3)",
                              10: "alcohol (vol.%)",
                              11: "quality"}
wine = wine.rename(columns=mapping_2)
names = list(wine.columns.values)
names.remove('quality')
target_name = 'quality'
drop = ['quality']


"""
Fitness function and representation (generating initial population)
"""

#fitness function
def fitness(x):
    if sum(x) == 0:
        return 0
    else:
        keep_me = []
        drop_me = drop.copy()
        for i in range(len(x)):
            if x[i] == 1:
                keep_me.append(names[i])
            else:
                drop_me.append(names[i])
        data = wine.drop(drop_me, axis = 1)
        kmeans = KMeans(n_clusters=7, random_state=0).fit(data)
        return metrics.completeness_score(wine["quality"], kmeans.labels_) + metrics.homogeneity_score(wine["quality"], kmeans.labels_)

#create initial population
from  random import randint
population = []
for j in range(popsize):
    individual = []
    i = 0
    while i < len(names):
        individual.append(randint(0,1))
        i += 1
    population.append(individual)

"""
The following functions are used to create the next generation of the population, and evaluate and record the best solutions so far
"""

#mutate function; changes one digit of the binary string in 5% of cases
def mutate(original):
    mutate = randint(1,100)
    if 95 < mutate:
        switch = randint(0,(len(names)-1))
        new = original.copy()
        if original[switch] == 0:
            new[switch] = 1
        else:
            new[switch] = 0
    else:
        new = original
    return new

    
#combine function; in 75% of cases, merges two parent strings to make two new baby strings that are added to the new population
#    in 25% of cases the two original parents are carried over without merging
def combine(pop, parent1, parent2):
    merge = randint(1,100)
    if merge <= 75:
        crossover = randint(1,(len(names) -2));
        new1 = parent1[:crossover] + parent2[crossover:] 
        new2 = parent2[:crossover] + parent1[crossover:] 
        new1 = mutate(new1)
        new2 = mutate(new2)
    else:
        new1 = mutate(parent1)
        new2 = mutate(parent2)
    pop.append(new1)
    pop.append(new2)    
    return pop

#get_best function; called once for each generation, and it keeps track of the best solution
def get_best(pop, high, best):
    all = []
    for element in pop:
        all.append(fitness(element))
    if max(all)>high:
        return(max(all), pop[all.index(max(all))])
    else:
        return best[len(best)-1]
        
#breed function; chooses new parents using the tournament selection and breeds them untill the new population is the same size as the previous  
def breed(population, best):
    newpop = []
    while len(newpop) < popsize:
        #choose 4 parents to compete
        comp1 = randint(0,(popsize -1))
        comp2 = randint(0,(popsize -1))
        comp3 = randint(0,(popsize -1))
        comp4 = randint(0,(popsize -1))
        #depending on competition outcomes, different pairings breed
        if fitness(population[comp1]) < fitness(population[comp2]):
            if fitness(population[comp3]) < fitness(population[comp4]):
                newpop = combine(newpop,population[comp2], population[comp4])
            else:
                newpop = combine(newpop, population[comp2], population[comp3])
        else:
            if fitness(population[comp3]) < fitness(population[comp4]):
                newpop = combine(newpop, population[comp1], population[comp4])
            else:
                newpop = combine(newpop,population[comp1], population[comp3])
    best.append(get_best(newpop, best[len(best)-1][0], best))
    return newpop


"""
This code starts the genetic algorithm
Until the solution is achieved or there have been the set number of max iterations, new populations are generated and tested
""" 

#for the decision tree with quality outcomes+
overall = []
#iterate the populations
high = 0 
best = [(0, "start")]
for i in range(iterate) :
    population = breed(population, best)
parameters = []
for i in range(len(names)):
    if best[len(best)-1][1][i] == 1:
        parameters.append(names[i])
        
print("The best attibutes to be used in the K-means clustering are", parameters)

data = wine.drop(["fixed acididty (g/dm^3)", "citric acid (g/dm^3)", "residual sugar (g/dm^3)", "chlorides (g/dm^3)", "free sulfur dioxide (mg/dm^3)","density(g/cm^3)", "pH","sulphates (g/dm^3)", "quality"], axis = 1)
kmeans = KMeans(n_clusters=7, random_state=0).fit(data)
print("completeness", metrics.completeness_score(wine["quality"], kmeans.labels_))
print("homogeneity" , metrics.homogeneity_score(wine["quality"], kmeans.labels_))


#determining the parameters for the k means clusters, measuring using silhouette score, when we don't test according to quality
def fitness(x):
    if sum(x) < 3:
        return 0
    else:
        keep_me = []
        drop_me = drop.copy()
        for i in range(len(x)):
            if x[i] == 1:
                keep_me.append(names[i])
            else:
                drop_me.append(names[i])
        data = wine.drop(drop_me, axis = 1)
        kmeans = KMeans(n_clusters=6, random_state=0).fit(data)
        return (metrics.silhouette_score(data, kmeans.labels_))

population = []
for j in range(popsize):
    individual = []
    i = 0
    while i < len(names):
        individual.append(randint(0,1))
        i += 1
    population.append(individual)

#for the decision tree with quality outcomes+
overall = []
#iterate the populations
high = 0 
best = [(0, "start")]
for i in range(iterate) :
    population = breed(population, best)
parameters = []
for i in range(len(names)):
    if best[len(best)-1][1][i] == 1:
        parameters.append(names[i])
        
print("The best attibutes to be used in the K-means clustering are", parameters)

data = wine.drop(["fixed acididty (g/dm^3)", "volatile acididty (g/dm^3)","citric acid (g/dm^3)", "total sulfur dioxide (mg/dm^3)","free sulfur dioxide (mg/dm^3)","density(g/cm^3)", "pH","sulphates (g/dm^3)", "quality"], axis = 1)
kmeans = KMeans(n_clusters=6, random_state=0).fit(data)
for element in parameters:
    print(element, ":")
    print("completeness", metrics.completeness_score(wine[element], kmeans.labels_))
    print("homogeneity" , metrics.homogeneity_score(wine[element], kmeans.labels_))

        