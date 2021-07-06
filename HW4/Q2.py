#write your code here
import random
  
class Ant(object): 
    def __init__(self, chromosome): 
        self.chromosome = chromosome  
        
    def fitness(self):
        chromosome_string = str(self.chromosome)
        
        left = int(chromosome_string[2:9],2)
        right = int(chromosome_string[9:],2)

        x = left + (right / (10 ** len(str(right))))

        if chromosome_string[1] == "1":
            x *= -1

        return abs((9*(x**5))-(194.7*(x**4))+(1680.1*(x**3))-(7227.94*(x**2))+(15501.2*x)-13257.2)

    
    def cross_over(self, parent2):

        dad = list(str(self.chromosome))
        mom = list(str(parent2.chromosome))
        
        r1 = random.randint(1,63)
        numbers = list(range(1, 64))
        numbers.remove(r1) 
        r2 = random.choice(numbers)      
    
        mom[min(r1, r2):max(r1, r2)] = dad[min(r1, r2):max(r1, r2)]
        child = dad


        if random.random() < 0.1:
            child = mutation(child)
        child = int("".join(child))
        return Ant(child) 
  

def initilize_chromosome(): 
    chromosome = '1'

    for i in range(63):
        chromosome += str(random.randint(0,1))

    chromosome = int(chromosome)
    return chromosome


def mutation(chromosome): 
    r = random.randint(1,63)

    if chromosome[r] == "0":
        chromosome[r] = "1"
    else:
        chromosome[r] = "0"

    return chromosome


def main():   
    generation = 0
    Ants = [] 
    number_of_population = 300


    for i in range(number_of_population): 
        Ants.append(Ant(initilize_chromosome())) 


    for i in range(250):

        generation += 1

        Ants = sorted(Ants, key = lambda x:x.fitness()) 
        if Ants[0].fitness() < 0.001:
            break

        else :
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
    

    chromosome_string = str(Ants[0].chromosome)
    left = int(chromosome_string[2:9],2)
    right = int(chromosome_string[9:],2)
    x = left + (right / (10 ** len(str(right))))
    if chromosome_string[1] == "1":
        x *= -1

    print("approximate Answer is : ", x)
    print("best chromosome: ", Ants[0].chromosome)
    print("generation: ", generation)
  
if __name__ == '__main__':
    main()


