import random
import numpy as np
from ga.fitness import fitness_function

# Constraints
GREEN_RANGE = (10, 60)
YELLOW_RANGE = (3, 6)
RED_RANGE = (10, 60)

def create_chromosome():
    return [
        random.randint(*GREEN_RANGE),
        random.randint(*YELLOW_RANGE),
        random.randint(*RED_RANGE)
    ]

def crossover(parent1, parent2):
    point = random.randint(1, 2)
    return parent1[:point] + parent2[point:]

def mutate(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        idx = random.randint(0, 2)
        chromosome[idx] += random.randint(-3, 3)
    return chromosome

def run_ga(avg_wait, avg_queue, pop_size, generations, mutation_rate):
    population = [create_chromosome() for _ in range(pop_size)]
    best_fitness_history = []

    for _ in range(generations):
        fitness_scores = [
            fitness_function(avg_wait, avg_queue) for _ in population
        ]

        best_fitness_history.append(min(fitness_scores))

        # Selection (elitism)
        selected = sorted(
            zip(population, fitness_scores),
            key=lambda x: x[1]
        )[:pop_size // 2]

        new_population = [p for p, _ in selected]

        while len(new_population) < pop_size:
            p1, p2 = random.sample(new_population, 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    best_solution = population[0]
    return best_solution, best_fitness_history
