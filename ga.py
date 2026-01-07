import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Traffic Light Optimization (GA)",
    layout="wide"
)

st.title("🚦 Traffic Light Optimization using Genetic Algorithm")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("traffic_dataset.csv")

df = load_data()

st.subheader("Traffic Dataset Preview")
st.dataframe(df.head())

# Extract dataset metrics
avg_wait = df["waiting_time"].mean()
avg_queue = df["vehicle_count"].mean()

# -------------------------------
# GA Parameters (Sidebar)
# -------------------------------
st.sidebar.header("Genetic Algorithm Parameters")

POP_SIZE = st.sidebar.slider("Population Size", 10, 100, 30)
GENERATIONS = st.sidebar.slider("Generations", 10, 200, 50)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)

# -------------------------------
# Traffic Signal Constraints
# -------------------------------
GREEN_RANGE = (10, 60)
YELLOW_RANGE = (3, 6)
RED_RANGE = (10, 60)

# -------------------------------
# Fitness Function
# -------------------------------
def fitness_function(avg_wait, vehicle_count):
    """
    Single-objective fitness
    Lower is better
    """
    return 0.6 * avg_wait + 0.4 * vehicle_count

# -------------------------------
# GA Operators
# -------------------------------
def create_chromosome():
    return [
        random.randint(*GREEN_RANGE),
        random.randint(*YELLOW_RANGE),
        random.randint(*RED_RANGE)
    ]

def crossover(parent1, parent2):
    point = random.randint(1, 2)
    return parent1[:point] + parent2[point:]

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, 2)
        chromosome[idx] += random.randint(-3, 3)
    return chromosome

# -------------------------------
# Run Genetic Algorithm
# -------------------------------
def run_ga():
    population = [create_chromosome() for _ in range(POP_SIZE)]
    best_fitness_history = []

    for _ in range(GENERATIONS):
        fitness_scores = [
            fitness_function(avg_wait, avg_queue)
            for _ in population
        ]

        best_fitness_history.append(min(fitness_scores))

        # Selection (elitism)
        selected = sorted(
            zip(population, fitness_scores),
            key=lambda x: x[1]
        )[:POP_SIZE // 2]

        new_population = [p for p, _ in selected]

        while len(new_population) < POP_SIZE:
            p1, p2 = random.sample(new_population, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return population[0], best_fitness_history

# -------------------------------
# Run Button
# -------------------------------
if st.button("Run Genetic Algorithm"):
    best_solution, fitness_history = run_ga()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimized Traffic Signal Timing")
        st.metric("Green Time (seconds)", best_solution[0])
        st.metric("Yellow Time (seconds)", best_solution[1])
        st.metric("Red Time (seconds)", best_solution[2])

    with col2:
        st.subheader("GA Convergence Curve")
        fig, ax = plt.subplots()
        ax.plot(fitness_history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title("Genetic Algorithm Convergence")
        st.pyplot(fig)

st.markdown("---")
st.caption("Traffic Light Optimization using Genetic Algorithm (GA)")
