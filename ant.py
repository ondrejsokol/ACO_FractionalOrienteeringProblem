import numpy as np
import random
from copy import deepcopy
import itertools
from itertools import compress
import time as tt
import sys

def my_print(text):
    sys.stdout.write(str(text))
    sys.stdout.flush()

class Graph():
    def __init__(self, nodes, distance, attraction, mandatory, W, min_sediment, alpha, beta, gamma, theta = 0.1):
        self.nodes = nodes
        self.distance = distance
        self.attraction = attraction
        self.mandatory = mandatory
        self.W = W
        self.min_sediment = 0
        assert distance.shape[1] == distance.shape[0]
        self.intensity_phi = np.full_like(distance, 1).astype('float64')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.iter = 0
        self.best_man = 0

def cycle_length(g, cycle):
    length = 0
    att = 0
    i = 0
    while i < len(cycle) - 1:
        length += g.distance[cycle[i]][cycle[i + 1]]
        i+=1
    length+= g.distance[cycle[i]][cycle[0]]
    for i in cycle:
        att += g.attraction[i]
    if length == 0:
        res = 9999
    else:
        res = -att / length
    return res

def cycle_length_parts(g, cycle):
    length = 0
    att = 0
    i = 0
    # print(cycle)
    while i < len(cycle) - 1:
        length += g.distance[cycle[i]][cycle[i + 1]]
        i+=1
    length+= g.distance[cycle[i]][cycle[0]]

    for i in cycle:
        att += g.attraction[i]
    
    if length == 0:
        res = 9999
    else:
        res = -att / length

    return res, att, length

def nearest_neighbour(g):
    man = deepcopy(g.mandatory)
    sol = []
    sol.append(g.mandatory[0])
    man.remove(sol[0])
    
    while len(man) > 1:
        index_min = np.argmin(g.distance[sol[len(sol)-1]][man])     
        if type(index_min) == list:
            index_min = index_min[0]
        sol.append(man[index_min])
        man.remove(man[index_min])   
    sol.append(man[0])
    return sol

def brute_force_permutations(g):
    best = 9999
    perm = list(itertools.permutations(g.mandatory))
    for i in range(len(perm)):
        cycle = list(perm[i])
        length = cycle_length(g, cycle) 
        if length < best:
            best = length
            best_cycle = cycle
    return best_cycle

def total_time(g,cycle):
    time = 0
    for i in range(1,len(cycle)):
        time = time + g.attraction[cycle[i]] + g.distance[cycle[i-1],cycle[i]] 
    time = time + g.distance[cycle[len(cycle)-1],0] + g.attraction[0]
    return time

def ant_colony_optimization(g, verbose=True, iterations = 100, ants_per_iteration = 50, computation_limit = 300):
    total_ants = 0
    best_cycle =  nearest_neighbour(g) 
    #best_cycle = brute_force_permutations(g)
    best_length = cycle_length(g, best_cycle)
    best_cycle_man = best_cycle
    best_visited_edges = np.asarray([[0 for _ in range(g.nodes)] for _ in range(g.nodes)]) 
    best_visited = np.asarray([0 for _ in range(g.nodes)])
    best_time = total_time(g, best_cycle)
    
    computation_time = tt.time()
    
    for iteration in range(iterations):
        cycles = [traverse_graph(g, 0) for _ in range(ants_per_iteration)]
        cycles.sort(key = lambda x: x[1])
        cycles = cycles[: ants_per_iteration]
        total_ants+=ants_per_iteration

        if best_cycle: #elitism
            cycles.append((best_cycle, best_length, best_cycle_man, best_visited, best_visited_edges, best_time))

        g.intensity_phi = (1 - g.gamma) * g.intensity_phi 
        for cycle, total_length, cycle_man, visited, visited_edges, time in cycles:
            if total_time(g, cycle) < g.W:  
                total_length = cycle_length(g, cycle)
                if total_length < best_length:
                    best_length = total_length
                    best_cycle = cycle
                    best_cycle_man = cycle_man
                    best_visited = visited
                    best_visited_edges = visited_edges
                    
                delta = 0
                if -total_length > g.best_man:
                    delta = -total_length - g.best_man
                    for j1 in range(g.nodes):
                        for j2 in range(g.nodes):
                            if visited_edges[j1][j2] == 1:
                                g.intensity_phi[j1][j2] += g.gamma * delta / ants_per_iteration
           
        # check if time exceeded
        if tt.time() - computation_time > computation_limit:
            break
        
        g.iter += 1      
    return best_cycle, g.intensity_phi


def traverse_graph(g, source_node = 0):
    time = 10e6
    cycle_man, total_length, visited, visited_edges, time = mandatory_route(g, source_node)         
    if time > g.W:
        my_print('mandatory route not feasible')
        return 0
    cycle, total_length, visited, visited_edges, time = improving_route(g, cycle_man, visited, visited_edges, total_length, time)
    total_length = cycle_length(g, cycle)
    assert len(list(set(cycle))) == len(cycle)
    return cycle, total_length, cycle_man, visited, visited_edges, time


def mandatory_route(g, source_node = 0):
    visited_edges = np.asarray([[0 for _ in range(g.nodes)] for _ in range(g.nodes)]) 
    visited = np.asarray([0 for _ in range(g.nodes)])
    visited[source_node] = 1
    
    manda = np.asarray([0 for _ in range(g.nodes)])
    for i in g.mandatory:
        manda[i] = 1
    manda[source_node] = 0
    
    cycle = [source_node]
    current = source_node
    total_length = 0
    next_node = source_node
    while sum(manda) > 0:
        next_node, visited, visited_edges = choose_next_mandatory(g, current, visited, visited_edges)         
        manda[next_node] = 0
        current = next_node
        cycle.append(current)
        
    total_length = cycle_length(g, cycle)
    time = total_time(g, cycle)
    
    if -total_length > g.best_man:
        g.best_man = -total_length
    return cycle, total_length, visited, visited_edges, time
    

def choose_next_mandatory(g,current,visited, visited_edges):
    weights_neighbors = []
    weights_values = []
    for node in g.mandatory:
        if visited[node] == 0:
           sediment = max( (g.intensity_phi[current][node])**g.alpha, g.min_sediment) #/ g.distance[current][node]**2
           weights_neighbors.append(node)
           weights_values.append(sediment)
    
    #with the probability of beta percent a node is selected randomly
    if sum(weights_values) == 0:
        weights_rand = 1
    else:
        weights_rand = sum(weights_values) / len(weights_values)
    for i in range(len(weights_values)):
        weights_values[i] =  (1 - g.beta) * weights_values[i] + g.beta * weights_rand
           
    next_node = random.choices(weights_neighbors, weights = weights_values)[0]
    
    visited[next_node] = 1
    visited_edges[current][next_node] = 1
    visited_edges[next_node][current] = 1
    return next_node, visited, visited_edges
    

def improving_route(g, cycle, visited, visited_edges, total_length, time):    
    counter = 1
    while time <= g.W:            
        visited_old = visited.copy()
        visited_edges_old = visited_edges.copy()
        
        old_node1, old_node2 = choose_next_to_replace(g, cycle)
        new_node, visited, visited_edges, break_check, time = choose_next_node(g, cycle, visited, visited_edges, time, old_node1, old_node2)      
                                        
        if break_check == 1:
            visited = visited_old
            visited_edges = visited_edges_old
            return cycle, total_length, visited, visited_edges, time
        else:        
            current = new_node
            pozice = 0
            for i in range(len(cycle)):
                if cycle[i] == old_node1:
                    pozice = i
            cycle.insert(pozice + 1, current)
            total_length = cycle_length(g, cycle)
        counter += 1
        
    return cycle, total_length, visited, visited_edges, time


def choose_next_to_replace(g, cycle):
    route_edges = [
        (cycle[i], cycle[(i + 1) % len(cycle)])
        for i in range(len(cycle))
    ]

    weights = []
    for node_x, node_y in route_edges:
        pheromone = max(
            g.intensity_phi[node_x][node_y],
            g.min_sediment
        )

        weight = (1.0 / pheromone) ** g.alpha
        weights.append(weight)

    average_weight = (
        sum(weights) / len(weights)
        if sum(weights) > 0
        else 1.0
    )

    weights = [
        (1.0 - g.beta) * weight
        + g.beta * average_weight
        for weight in weights
    ]

    old_node1, old_node2 = random.choices(
        route_edges,
        weights=weights,
        k=1
    )[0]

    return old_node1, old_node2


def choose_next_node(g, cycle, visited, visited_edges, time, old_node1, old_node2):
    weights_neighbors = []
    weights_valuesX = []
    weights_valuesY = []
    break_valuesX = []
    break_valuesY = []
    not_visited = [not elem for elem in visited]
    nodes_to_explore = list(compress(list(range(g.nodes)), not_visited))
    res, att, length = cycle_length_parts(g, cycle)
    length = length - g.distance[old_node1][old_node2]
    att = float(att)
       
    for node in nodes_to_explore:          
        sedimentX = (( ( g.intensity_phi[old_node1][node])**g.alpha) * ( ( g.intensity_phi[old_node2][node])**g.alpha) * ((1/ (g.intensity_phi[old_node1][old_node2]))**g.alpha))  
        sedimentY =   (att + float(g.attraction[node])) / (length + g.distance[old_node1, node] + g.distance[node, old_node2] ) 
        new_time = time  + g.distance[old_node1, node] + g.distance[node, old_node2] + g.attraction[node] - g.distance[old_node1, old_node2]
        if new_time <= g.W:
            if (visited_edges[old_node1][node] == 0 or visited_edges[old_node2][node] == 0):
                weights_neighbors.append(node)
                weights_valuesX.append(sedimentX)
                weights_valuesY.append(sedimentY)
            else:
                break_valuesX.append(sedimentX) 
                break_valuesY.append(sedimentY)    
    if len(weights_neighbors) == 0:
        return 0, visited, visited_edges, 1, time    
    
    break_probX = ( (1 +  g.intensity_phi[old_node1][old_node2])**g.alpha)  /  (1 + g.intensity_phi[old_node1][old_node2])**g.alpha 
    break_probX = break_probX / (break_probX + sum(weights_valuesX) )        
    break_prob =  break_probX                                                        
    
    if random.uniform(0, 1) < break_prob:
        return 0, visited, visited_edges, 1, time
            
    if random.uniform(0, 1) < g.theta:    
        #with the probability of beta percent a node is selected randomly
        if sum(weights_valuesX) == 0:
            weights_rand = 1
        else:
            weights_rand = sum(weights_valuesX) / len(weights_valuesX)
        for i in range(len(weights_valuesX)):
            weights_valuesX[i] =  (1 - g.beta) * weights_valuesX[i] + g.beta * weights_rand
            
        new_node = random.choices(weights_neighbors, weights = weights_valuesX)[0]  
    else:           
        new_node_index = weights_valuesY.index(max(weights_valuesY))
        new_node = weights_neighbors[new_node_index]
        
    visited[new_node] = 1
    visited_edges[old_node1][new_node] = 1
    visited_edges[old_node2][new_node] = 1
    visited_edges[new_node][old_node1] = 1
    visited_edges[new_node][old_node2] = 1
    time = time  + g.distance[old_node1, new_node] + g.distance[new_node, old_node2] + g.attraction[new_node] - g.distance[old_node1, old_node2]
    
    if time > g.W:
        my_print('error, route not feasible')  
        return 0
    
    return new_node, visited, visited_edges, 0, time



