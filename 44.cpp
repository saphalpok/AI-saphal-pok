import random

# Define the parameters
POP_SIZE = 20
GENES = 11  # 5 for integer part, 6 for residual part
MAX_GEN = 100
MUTATION_RATE = 0.01

def decode_chromosome(chromosome):
    # Convert binary to integer, handling sign separately
    sign = chromosome[0]
    integer_part = chromosome[1:6]
    residual_part = chromosome[6:]
    
    integer = int("".join(map(str, integer_part)), 2)
    residual = int("".join(map(str, residual_part)), 2) / (2 ** len(residual_part))
    
    if sign == 1:
        x = integer + residual
    else:
        x = -(integer + residual)
    
    return x

def fitness(chromosome):
    x = decode_chromosome(chromosome)
    return -(x**2 - 4*x + 3)  # We want to maximize the function

def selection(population):
    selected = random.choices(
        population,
        weights=[fitness(chromosome) for chromosome in population],
        k=2
    )
    return selected

def crossover(parent1, parent2):
    point = random.randint(1, GENES - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def mutate(chromosome):
    for i in range(GENES):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def generate_population():
    return [[random.randint(0, 1) for _ in range(GENES)] for _ in range(POP_SIZE)]

def genetic_algorithm():
    population = generate_population()
    
    for generation in range(MAX_GEN):
        new_population = []
        
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = selection(population)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            new_population.extend([offspring1, offspring2])
        
        population = new_population
        
        if generation % 10 == 0:
            best = max(population, key=fitness)
            print(f"Generation {generation}: Best fitness = {fitness(best)}")
    
    best = max(population, key=fitness)
    best_solution = decode_chromosome(best)
    print(f"Best solution: x = {best_solution}, fitness = {fitness(best)}")

# Run the genetic algorithm
genetic_algorithm()
import random

# Define the parameters
POP_SIZE = 20
GENES = 11  # 5 for integer part, 6 for residual part
MAX_GEN = 100
MUTATION_RATE = 0.01

def decode_chromosome(chromosome):
    # Convert binary to integer, handling sign separately
    sign = chromosome[0]
    integer_part = chromosome[1:6]
    residual_part = chromosome[6:]
    
    integer = int("".join(map(str, integer_part)), 2)
    residual = int("".join(map(str, residual_part)), 2) / (2 ** len(residual_part))
    
    if sign == 1:
        x = integer + residual
    else:
        x = -(integer + residual)
    
    return x

def fitness(chromosome):
    x = decode_chromosome(chromosome)
    return -(x**2 - 4*x + 3)  # We want to maximize the function

def selection(population):
    selected = random.choices(
        population,
        weights=[fitness(chromosome) for chromosome in population],
        k=2
    )
    return selected

def crossover(parent1, parent2):
    point = random.randint(1, GENES - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

def mutate(chromosome):
    for i in range(GENES):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def generate_population():
    return [[random.randint(0, 1) for _ in range(GENES)] for _ in range(POP_SIZE)]

def genetic_algorithm():
    population = generate_population()
    
    for generation in range(MAX_GEN):
        new_population = []
        
        for _ in range(POP_SIZE // 2):
            parent1, parent2 = selection(population)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            new_population.extend([offspring1, offspring2])
        
        population = new_population
        
        if generation % 10 == 0:
            best = max(population, key=fitness)
            print(f"Generation {generation}: Best fitness = {fitness(best)}")
    
    best = max(population, key=fitness)
    best_solution = decode_chromosome(best)
    print(f"Best solution: x = {best_solution}, fitness = {fitness(best)}")

# Run the genetic algorithm
genetic_algorithm()
