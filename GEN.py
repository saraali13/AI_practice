import random

# Fitness function (example: maximizing f(x) = 2x² - 1)
def calculate_fitness(individual):
    x = int("".join(map(str, individual)), 2)  # Convert binary string to integer
    return 2 * x ** 2 - 1

# Generate random individual (6-bit binary)
def create_random_individual():
    return [random.randint(0, 1) for _ in range(6)]

# Tournament Selection for Parents
def select_parents(population, fitness_scores):
    selected = []
    for _ in range(len(population) // 2):
        contenders = random.sample(list(zip(population, fitness_scores)), 3)
        selected.append(max(contenders, key=lambda x: x[1])[0])
    return selected

# Uniform Crossover
def crossover(parent1, parent2):
    return [random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)]

# Adaptive Mutation
def mutate(individual, generation, max_generations):
    mutation_rate = 0.1 if generation < max_generations // 2 else 0.2
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip bit
    return individual

# Main Genetic Algorithm Function
def genetic_algorithm(pop_size=10, generations=100):
    population = [create_random_individual() for _ in range(pop_size)]
    best_fitness = float('-inf')
    best_individual = None

    for gen in range(generations):
        fitness_scores = [calculate_fitness(ind) for ind in population]
        best_gen_fit = max(fitness_scores)

        if best_gen_fit > best_fitness:
            best_fitness = best_gen_fit
            best_individual = population[fitness_scores.index(best_gen_fit)]

        print(f"Generation {gen} | Best Fitness: {best_fitness}")
        if best_fitness == 1983:  # Maximum value for f(x) = 2x² - 1 when x = 31
            break

        # Selection
        parents = select_parents(population, fitness_scores)

        # Crossover
        new_population = [crossover(random.choice(parents), random.choice(parents)) for _ in range(pop_size)]

        # Mutation
        new_population = [mutate(ind, gen, generations) for ind in new_population]

        population = new_population

    return best_individual, best_fitness

# Run the Genetic Algorithm
solution, fitness = genetic_algorithm()
print("Best Solution (Binary):", "".join(map(str, solution)))
print("Best Fitness Value:", fitness)
