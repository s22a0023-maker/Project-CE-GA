import streamlit as st
import pandas as pd
import numpy as np
import random
import time

st.title("Traffic Light Optimization using Genetic Algorithm")

# Load dataset
data = pd.read_csv("traffic_dataset.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

#------
#------
def fitness_function(green_times, traffic_data):
    avg_wait = traffic_data["waiting_time"].mean()
    queue = traffic_data["vehicle_count"].mean()

    penalty = 0
    if sum(green_times) > 180:
        penalty += 100

    fitness = avg_wait + queue + penalty
    return fitness
#-------
#-------

def initialize_population(pop_size, phases):
    return [np.random.randint(10, 60, phases).tolist() for _ in range(pop_size)]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    return parent1[:point] + parent2[point:]

def mutation(chromosome, rate=0.1):
    for i in range(len(chromosome)):
        if random.random() < rate:
            chromosome[i] = random.randint(10, 60)
    return chromosome


#-------
#-------
def genetic_algorithm(generations, pop_size, phases):
    population = initialize_population(pop_size, phases)
    best_fitness = []
    
    for gen in range(generations):
        scored = [(fitness_function(ind, data), ind) for ind in population]
        scored.sort(key=lambda x: x[0])
        population = [x[1] for x in scored[:pop_size//2]]

        while len(population) < pop_size:
            p1, p2 = random.sample(population[:10], 2)
            child = crossover(p1, p2)
            population.append(mutation(child))

        best_fitness.append(scored[0][0])

    return best_fitness, scored[0][1]

#------
#------
st.subheader("GA Parameter Settings")

generations = st.slider("Generations", 10, 200, 50)
population = st.slider("Population Size", 10, 100, 30)

if st.button("Run Genetic Algorithm"):
    start = time.time()
    fitness_curve, best_solution = genetic_algorithm(generations, population, 4)
    end = time.time()

    st.success("Optimization Completed")
    st.write("Best Green Time Plan:", best_solution)
    st.write("Execution Time (s):", round(end-start, 3))

    st.line_chart(fitness_curve)
#------
#------

def multi_objective_fitness(green_times, traffic_data):
    wait = traffic_data["waiting_time"].mean()
    queue = traffic_data["vehicle_count"].mean()
    cycle = sum(green_times)

    return (0.5 * wait) + (0.3 * queue) + (0.2 * cycle)

#-----
#------

st.subheader("Objective Trade-Off Analysis")

results = []
for _ in range(20):
    sol = np.random.randint(10, 60, 4)
    results.append({
        "Waiting Time": data["waiting_time"].mean(),
        "Queue Length": data["vehicle_count"].mean(),
        "Cycle Time": sum(sol)
    })

df_results = pd.DataFrame(results)
st.scatter_chart(df_results)

