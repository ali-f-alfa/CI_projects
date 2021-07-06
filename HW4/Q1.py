import random 
import numpy as np

aa = 99999
Edges = [[0,12,10,aa,aa,aa,12],
           [12,0,8,12,aa,aa,aa],
           [10,8,0,11,3,aa,9],
           [aa,12,11,0,11,10,aa],
           [aa,aa,3,11,0,6,7],
           [aa,aa,aa,10,6,0,9],
           [12,aa,9,aa,7,9,0]]

class Ant(): 
    def __init__(self, chromosome): 
        self.chromosome = chromosome  
  
    def fitness(self): 
        sum = 0
        for i in range(7):
            sum += Edges[self.chromosome[i]][self.chromosome[i+1]]
        return sum
  
    def cross_over(self, parent2):     
        dad = []
        mom = []
        
        r1 = random.randint(1,6)
        numbers = list(range(1, 7))
        numbers.remove(r1) 
        r2 = random.choice(numbers)   

        for i in range(min(r1, r2), max(r1, r2)):
            dad.append(self.chromosome[i])
        for i in range(len(parent2.chromosome)):
            if (parent2.chromosome[i] not in dad) and len(mom)< min(r1, r2) :
                mom.append(parent2.chromosome[i])
        
        child = mom + dad

        for i in range(len(parent2.chromosome)):
            if parent2.chromosome[i] not in child:
                child.append(parent2.chromosome[i])
        child.append(0)

        # mutation
        if random.random() < 0.1:
            child = mutation(child)
            
        return Ant(child) 
  


def initilize_chromosome():        
    x = [0] 
    x.extend(random.sample(range(1,7), 6))
    x.append(0)
    return x
   
def mutation(chromosome): 
    r1 = random.randint(1,6)
    numbers = list(range(1, 7))
    numbers.remove(r1) 
    r2 = random.choice(numbers)     
    
    chromosome[r1],chromosome[r2] = chromosome[r2],chromosome[r1]
    return chromosome


if __name__ == '__main__':

    number_of_population = 35
    generation = 0
    Ants = [] 
  
    for i in range(number_of_population): 
        Ants.append(Ant(initilize_chromosome())) 

    for i in range(1000):
        generation += 1
        Ants = sorted(Ants, key = lambda x:x.fitness()) 
  
        children = [] 

        p = int(number_of_population*0.5) 
        children.extend(Ants[:p]) 
  
        p = int(number_of_population*0.7) 
        for i in range(p): 
            parent1 = random.choice(Ants[:50]) 
            parent2 = random.choice(Ants[:50]) 
            child = parent1.cross_over(parent2) 
            children.append(child) 
  
        Ants = children 
  
    answer = Ants[0]
    print("answer path :" , [x + 1 for x in answer.chromosome])
    print( "path length :", answer.fitness())
    print("Generation:", generation)


