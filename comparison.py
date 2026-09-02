import tsplib95
import sys
import os
import networkx
import random
import math
from ant import *
import numpy as np
import pandas as pd
from copy import deepcopy
import itertools
from itertools import compress
from gurobi_opti import *
import re
from datetime import datetime
import time
import multiprocessing as mp
from functools import partial
from gurobi_opti_DFJ import *

def generateOP_NN(n, avg_edge, g):
    random.seed(n)
    np.random.seed(n) 
    dur = np.round(np.random.exponential(1, n) * avg_edge / 4, 2)
    man = [0]
    man = man + random.sample(range(1, n), min(n // 2, 1 + np.random.poisson(1) )) 
    man.sort()    
    g.mandatory = man
    g.attraction = dur
    nn_cycle = nearest_neighbour(g)
    nn_time = total_time(g, nn_cycle)
    # print(cycle_length(g, nn_cycle))
    tim = math.ceil(nn_time * (np.random.uniform(low = 1.3, high = 3) ))
    return dur, man, tim


def run_ants(soubory, alpha, beta, gamma, theta):
    res = []
    res_geom = 1
    for file in soubory:   
        random.seed(1)
        np.random.seed(1) 
        problem = tsplib95.load(file)    
        graph = problem.get_graph()
        d = networkx.to_numpy_matrix(graph)
        n, n = d.shape   
        avg_edge = np.matrix.sum(d) / (n * n)
        
        g_gen = Graph(n, distance = np.array(d), attraction = [], mandatory = [0], W = 1, min_sediment = 0, alpha = alpha, beta = beta, gamma = gamma, theta = theta)
        dur, man, tim = generateOP_NN(n, avg_edge, g_gen)
        
        np.fill_diagonal(d, +1e6)

        iterations = 200
        ants = 100
        dur = pd.DataFrame(dur)
        d = pd.DataFrame(d)
        
        d = np.array(d)
        dur = np.array(dur)
        g = Graph(n, distance = d, attraction = dur, mandatory = man, min_sediment = 0, W = tim, alpha = alpha, beta = beta, gamma = gamma, theta = theta)
        zac = time.time()
        solution, feromony = ant_colony_optimization(g, iterations = iterations, ants_per_iteration = ants)
        kon = time.time() - zac
        sol = cycle_length(g, solution)[0]
        
        res.append([file, iterations, ants, dur, man, tim, alpha, beta, gamma, theta, solution, sol, int(kon)] )
       
        res_geom = res_geom * sol * (-1)
    return res_geom * (-1), res


def run_gurobi(soubory):
    now = datetime.now()
    print(now.strftime("%d_%m_%Y_%H_%M"))
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    res = []
    for file in soubory:
        random.seed(1)
        np.random.seed(1) 
        problem = tsplib95.load(file)    
        graph = problem.get_graph()
        d = networkx.to_numpy_matrix(graph)
        n, n = d.shape   
        avg_edge = np.matrix.sum(d) / (n * n)
        
        g_gen = Graph(n, distance = np.array(d), attraction = [], mandatory = [0], W = 1, min_sediment = 0, alpha = 1, beta = 1, gamma = 1)
        dur, man, tim = generateOP_NN(n, avg_edge, g_gen)        
        
        dur = pd.DataFrame(dur)
        d = pd.DataFrame(d)
        d = np.array(d)
        dur = np.array(dur)        
        gurobi_solution, objVal , gap, vystup  = gurobi_ant_dfj(d = d, c = dur, M = man, W = tim, time_limit=300)
        res.append([file, dur, man, tim, gurobi_solution, objVal , gap])
        
    return res



directory = 'INSERT directory with TSPlib instances'
files = []
for file in directory:
    if file.endswith(".tsp"):
        res = int(re.search(r'\d+', file).group(0))
        files.append(file)


gur = run_gurobi(files)
aco = run_ants(files)


