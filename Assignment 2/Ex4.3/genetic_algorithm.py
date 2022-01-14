from search import *
from math import floor
import random
import numpy as np
import matplotlib.pyplot as plt

# Needed to hide warnings in the matplotlib sections
import warnings
warnings.filterwarnings("ignore")


# use cities from the Romania map as cities for TSP problem
all_cities_name = np.array(['Arad','Bucharest','Craiova','Drobeta','Eforie','Fagaras','Giurgiu','Hirsova','Iasi',
                    'Lugoj','Mehadia','Neamt','Oradea','Pitesti','Rimnicu','Sibiu','Timisoara','Urziceni','Vaslui','Zerind'])

all_cities_location = np.array([91, 492, 400, 327, 253, 288, 165, 299, 562, 293,
                        305, 449,375, 270,534, 350,473, 506,165, 379,
                        168, 339,406, 537,131, 571,320, 368,233, 410,
                        207, 457,94, 410,456, 350,509, 444,108, 531]).reshape(20,2)


# genetic algorithm       
class Gena_TSP(object):
    def __init__(self,data,maxgen=200,size_pop=200,cross_prob=0.9,pmuta_prob=0.01,select_prob=0.8):
        # the max number of iterations
        self.maxgen = maxgen 
        # the initial population
        self.size_pop = size_pop  
        # the probability of cross
        self.cross_prob = cross_prob 
        # the probability of mutation
        self.pmuta_prob = pmuta_prob 
        # the probability of select
        self.select_prob = select_prob 
        
        # cities
        self.data = data   
        # the number of cities
        self.num =len(data) 
        
        # the distance between two cities
        self.matrix_distance = self.matrix_dis() 
        
        # the number of next generation selected
        self.select_num = max(floor(self.size_pop*self.select_prob+0.5),2) 
       
        # the initialization of populations of generation and next generation
        self.chrom = np.array([0]*self.size_pop*self.num).reshape(self.size_pop,self.num)
        self.sub_sel = np.array([0]*self.select_num*self.num).reshape(self.select_num,self.num)
       
        # every city's distance
        self.fitness = np.zeros(self.size_pop)
        
        # save for every step
        self.best_fit = []
        self.best_path= []
    
    # calculate the distance between cities
    def matrix_dis(self):
        res = np.zeros((self.num,self.num))
        for i in range(self.num):
            for j in range(i+1,self.num):
                res[i,j] = np.linalg.norm(self.data[i,:]-self.data[j,:])
                res[j,i] = res[i,j]
        return res

    # generate a random population
    def rand_chrom(self):
        rand_ch = np.array(range(self.num))
        for i in range(self.size_pop):
            np.random.shuffle(rand_ch)
            self.chrom[i,:]= rand_ch
            self.fitness[i] = self.comp_fit(rand_ch)

    # calculate the distance from one city
    def comp_fit(self, one_path):
        res = 0
        for i in range(self.num-1):
            res += self.matrix_distance[one_path[i],one_path[i+1]]
        res += self.matrix_distance[one_path[-1],one_path[0]]
        return res

    # print the path
    def out_path(self, one_path):
        res = str(all_cities_name[one_path[0]])+' --> '
        for i in range(1, self.num):
            res += str(all_cities_name[one_path[i]])+' --> '
        res += str(all_cities_name[one_path[0]])+'\n'
        print(res)

    # select the next genaration
    def select_sub(self):
        fit = 1./(self.fitness) 
        cumsum_fit = np.cumsum(fit)
        pick = cumsum_fit[-1]/self.select_num*(np.random.rand()+np.array(range(self.select_num)))
        i,j = 0,0
        index = []
        while i<self.size_pop and j<self.select_num:
            if cumsum_fit[i]>= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index,:]

    # cross
    def cross_sub(self):
        if self.select_num%2 == 0:
            num = range(0,self.select_num,2)
        else:
            num = range(0,self.select_num-1,2)
        for i in num:
            if self.cross_prob>=np.random.rand():
                self.sub_sel[i,:],self.sub_sel[i+1,:] = self.intercross(self.sub_sel[i,:],self.sub_sel[i+1,:])
                
    def intercross(self,ind_a,ind_b):
        r1 = np.random.randint(self.num)
        r2 = np.random.randint(self.num)
        while r2 == r1:
            r2 = np.random.randint(self.num)
        left,right = min(r1,r2),max(r1,r2)
        ind_a1 = ind_a.copy()
        ind_b1 = ind_b.copy()
        for i in range(left,right+1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i] 
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a==ind_a[i])
            y = np.argwhere(ind_b==ind_b[i])
            if len(x) == 2:
                ind_a[x[x!=i]] = ind_a2[i]
            if len(y) == 2:
                ind_b[y[y!=i]] = ind_b2[i]
        return ind_a,ind_b

    # pick a gene in x to mutate and a gene from the gene pool to replace it with
    def mutation_sub(self):
        for i in range(self.select_num):
            if np.random.rand() <= self.cross_prob:
                r1 = np.random.randint(self.num)
                r2 = np.random.randint(self.num)
                while r2 == r1:
                    r2 = np.random.randint(self.num)
                self.sub_sel[i,[r1,r2]] = self.sub_sel[i,[r2,r1]]
    # reverse
    def reverse_sub(self):
        for i in range(self.select_num):
            r1 = np.random.randint(self.num)
            r2 = np.random.randint(self.num)
            while r2 == r1:
                r2 = np.random.randint(self.num)
            left,right = min(r1,r2),max(r1,r2)
            sel = self.sub_sel[i,:].copy()
            
            sel[left:right+1] = self.sub_sel[i,left:right+1][::-1]
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i,:]):
                self.sub_sel[i,:] = sel

    # insert the new generation to the origin one
    def reins(self):
        index = np.argsort(self.fitness)[::-1]
        self.chrom[index[:self.select_num],:] = self.sub_sel

# main process
def main(data):
    # data is the city information
    Path_short = Gena_TSP(data) 
    Path_short.rand_chrom()  

    # iterations
    for i in range(Path_short.maxgen):
        Path_short.select_sub()   
        Path_short.cross_sub()    
        Path_short.mutation_sub() 
        Path_short.reverse_sub()  
        Path_short.reins()        

        # calculate the new path
        for j in range(Path_short.size_pop):
            Path_short.fitness[j] = Path_short.comp_fit(Path_short.chrom[j,:])
         
        # print the best path every 50 iterations
        index = Path_short.fitness.argmin()
        if (i+1)%50 == 0:
            print('The shorest path after '+str(i+1)+' iterations:'+str( Path_short.fitness[index]))
            print('The best path after '+str(i+1)+' iterations:')
            # print the best path every step
            Path_short.out_path(Path_short.chrom[index,:])
        
        # save best path every step
        Path_short.best_fit.append(Path_short.fitness[index])
        Path_short.best_path.append(Path_short.chrom[index,:])
    return Path_short  

# create individuals with random genes and return th epopulation when done
def init_population(pop_number, gene_pool, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""
    g = len(gene_pool)
    population = []
    for i in range(pop_number):
        new_individual = [gene_pool[random.randrange(0, g)] for j in range(state_length)]
        population.append(new_individual)

    return population


main(all_cities_location)

