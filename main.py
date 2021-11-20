from tqdm import tqdm, tqdm_notebook
from memory_profiler import memory_usage
import util
import time
import numpy as np
import pandas as pd
from ga import GA_TSP
from aco import ACO_TSP
from aco_tsp import SolveTSPUsingACO


TEST_TIMES = 10
TEST_SIZE = 100
result_history, result_last_state, result_obj_function_value, result_elapsed_time, result_memory = [], [], [], [], []

def ga():
    # time start 
    start = time.time()
    data = util.read_data(f'./tsp{TEST_SIZE}.txt')
    data = np.array(data)
    # GA_TSP
    tsp = GA_TSP()
    tsp.set_loc(data)
    tsp.solve()
    # ACO_TSP
#     tsp = ACO_TSP()
#     tsp.set_loc(data)
#     tsp.solve(n_agent=1000)
    tsp.plot(tsp.result)
    # time stop
    elapsed_time = time.time() - start
    elapsed_time = round(elapsed_time,2)
    return elapsed_time

def aco(index=0): 
    start = time.time()
    _colony_size = 10
    _steps = 250
    _nodes = util.read_data(f'./tsp{TEST_SIZE}.txt')
    max_min = SolveTSPUsingACO(mode='MaxMin', colony_size=_colony_size, steps=_steps, nodes=_nodes)
    max_min.run()
    max_min.plot(name=f'./aco_results/{TEST_SIZE}/aco_{index}')
    elapsed_time = time.time() - start
    elapsed_time = round(elapsed_time, 2)
    fitness = max_min.global_best_distance

    return elapsed_time, fitness

if __name__ == "__main__":
    for i in tqdm(range(TEST_TIMES)):

        # elapsed_time = ga()
        elapsed_time, fitness = aco(i)
        memory_out = memory_usage((aco,()),interval=1)
        avg_memory_out = sum(memory_out) / len(memory_out)
        result_elapsed_time.append(elapsed_time)
        result_memory.append(avg_memory_out)
        result_obj_function_value.append(fitness)
        
    
    max_elapsed_time, avg_elapsed_time, min_elapsed_time  = max(result_elapsed_time), sum(result_elapsed_time) / len(result_elapsed_time), min(result_elapsed_time)
    max_memory, avg_memory, min_memory  = max(result_memory), sum(result_memory) / len(result_memory), min(result_memory)
    max_fitness, avg_fitness, min_fitness = max(result_obj_function_value), sum(result_obj_function_value)/len(result_obj_function_value), min(result_obj_function_value)
    # Show the results
    df = pd.DataFrame()
    df['max'] = 0
    df['avg'] = 0 
    df['min'] = 0 
    df.loc['Memory'] = [max_memory, avg_memory, min_memory ]
    df.loc['Elapsed_time'] = [max_elapsed_time, avg_elapsed_time, min_elapsed_time]
    df.loc['Fitness'] = [max_fitness, avg_fitness, min_fitness]
    df.to_csv(f'./aco_results/{TEST_SIZE}/result_aco_{TEST_SIZE}.csv')