import random
import numpy

mem_cells_count = 64
first_gen_mem_cells_count = 32
max_steps = 500
elitism = 0.02  # percent of new individuals made by elitism
new_individuals = 0.15  # percent of new individuals made for new generation
mutation_rate = 0.05  # probability of mutation
tournament = 0.2  # number of individuals in tournament

number_of_gens = 200
number_of_individuals = 80

#TODO : implementovat parser pre txt vstupy

class Individual:
    def __init__(self, path, memory_cells, treasures_found, fitness_func):
        self.path = path
        self.memory_cells = memory_cells
        self.treasures_found = treasures_found
        self.fitness_func = fitness_func

    def set_path(self, path):
        self.path = path

    def set_memory_cells(self, memory_cells):
        self.memory_cells = memory_cells

    def set_treasures_found(self, treasures_found):
        self.treasures_found = treasures_found

    def set_fitness_func(self, fitness_func):
        self.fitness_func = fitness_func


map_size = (7,7)
all_treasures = {(1,4), (2,2), (3,6), (4,1), (5,4)}
start_position = (6,3)
best_individual = Individual([], [], set(), 0)
averages = []
bests = []
individuals = []


def spawn_generation():
    for i in range(number_of_individuals):
        individuals.append(spawn_individual())
        for j in range(mem_cells_count):
            if j <= first_gen_mem_cells_count:
                individuals[i].memory_cells.append(numpy.uint8(random.randrange(256)))
            else:
                individuals[i].memory_cells.append(numpy.uint8(0))


def spawn_individual():
    path = []  # Initialize path as an empty list
    memory_cells = [numpy.uint8(random.randint(0, 255)) for _ in range(mem_cells_count)]  # Create memory_cells
    treasures_found = set()  # Initialize treasures_found as an empty set
    fitness_func = 0  # Initialize fitness_func to 0

    # Create the new individual using the Individual class
    new_individual = Individual(path, memory_cells, treasures_found, fitness_func)

    return new_individual

def return_move(bits):
    if bits == "00":
        return "H"
    elif bits == "01":
        return "D"
    elif bits == "10":
        return "P"
    elif bits == "11":
        return "L"
    return None


def virtual_machine(individual):
    program_counter = 0
    index = 0
    instructions = individual.memory_cells

    while program_counter <= max_steps and 0 <= index < mem_cells_count:
        operation = format(instructions[index], '08b')[0:2]
        address = int(format(instructions[index], '08b')[2:], 2)

        if operation == "00":  # Increment
            individual.memory_cells[address] += 1
            individual.memory_cells[address] = numpy.uint8(individual.memory_cells[address])

        elif operation == "01":  # Decrement
            individual.memory_cells[address] -= 1
            individual.memory_cells[address] = numpy.uint8(individual.memory_cells[address])

        elif operation == "10":  # Jump
            index = address

        elif operation == "11":  # Write uses last 2 bits
            move = format(instructions[index], '08b')[6:]
            individual.path.append(return_move(move))

        program_counter += 1
        index += 1

def found_treasures(individual):
    row, column = start_position
    counter = 0
    updated_path = []

    for move in individual.path:
        if move == "H":
            row -= 1
        elif move == "D":
            row += 1
        elif move == "P":
            column += 1
        elif move == "L":
            column -= 1

        if row < 0 or row >= map_size[0] or column < 0 or column >= map_size[1]:
            # Check if the individual has moved outside the map
            break  # Exit the loop

        curr_pos = (row, column)
        if curr_pos in all_treasures and curr_pos not in individual.treasures_found:
            # Check for treasure and add it to the individual's treasures
            individual.treasures_found.add(curr_pos)

        updated_path.append(move)

    individual.path = updated_path  # Update the path

    if len(individual.treasures_found) == len(all_treasures):
        # All treasures found, searching ends
        return


def calculate_fitness(individual):
    if not individual.path:
        return 0  # Return 0 fitness for individuals with no path

    treasures_found = len(individual.treasures_found)
    total_steps = len(individual.path)

    # You can adjust these weights to prioritize different aspects of fitness
    treasures_weight = 10
    steps_weight = -0.001  # Penalize more steps

    # Calculate fitness score (a higher score is better)
    fitness = (treasures_weight * treasures_found) + (steps_weight * total_steps)

    print(fitness)
    if fitness == 49.999:
        print(individual.path)
        print(individual.treasures_found)

    return fitness


def crossover(parent1, parent2):
    # Perform single-point crossover
    crossover_point = random.randint(1, mem_cells_count - 1)

    child_memory_cells = parent1.memory_cells[:crossover_point] + parent2.memory_cells[crossover_point:]

    return child_memory_cells


def roulette_selection(sorted_gen, new_generation):
    start_index = int((elitism + new_individuals) * number_of_individuals)
    weights = [ind.fitness_func for ind in sorted_gen]

    for i in range(start_index, number_of_individuals):
        mom = random.choices(sorted_gen, weights=weights)[0]
        dad = random.choices(sorted_gen, weights=weights)[0]

        # Ensure that mom and dad are different individuals
        while mom == dad:
            dad = random.choices(sorted_gen, weights=weights)[0]

        # Create a new offspring Individual object and perform crossover on memory_cells
        offspring_memory_cells = crossover(mom, dad)
        offspring = Individual([], offspring_memory_cells, set(), 0)
        new_generation[i] = offspring


def tournament_selection(sorted_gen, new_generation):
    start_index = int((elitism + new_individuals) * number_of_individuals)
    num_tournament = int(tournament * number_of_individuals)
    for i in range(start_index, number_of_individuals):
        mom = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x.fitness_func)[0]
        dad = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x.fitness_func)[0]

        while numpy.array_equal(dad.memory_cells, mom.memory_cells):
            dad = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x.fitness_func)[0]

        # Create a new offspring Individual object and perform crossover on memory_cells
        offspring_memory_cells = crossover(mom, dad)
        offspring = Individual([], offspring_memory_cells, set(), 0)
        new_generation[i] = offspring


def single_point_crossover(parent1, parent2):
    # Perform single-point crossover
    crossover_point = random.randint(1, len(parent1.memory_cells) - 1)
    offspring = Individual([], [], set(), 0)

    offspring.memory_cells = parent1.memory_cells[:crossover_point] + parent2.memory_cells[crossover_point:]

    return offspring


def mutate_invert(individual):
    mutated_memory_cells = list(individual.memory_cells)  # Create a new list to hold the mutated memory cells

    for index, value in enumerate(individual.memory_cells):
        if random.random() < mutation_rate:
            shift = random.randint(0, 7)
            mutated_memory_cells[index] = numpy.uint8(value ^ (1 << shift))

    mutated_individual = Individual(individual.path, mutated_memory_cells, individual.treasures_found, individual.fitness_func)
    return mutated_individual


def mutation(new_generation):
    start_index = int((elitism + new_individuals) * number_of_individuals)
    for i in range(start_index, number_of_individuals):
        if i in new_generation:  # Check if the index exists in the dictionary
            individual = new_generation[i]  # Get the individual by index
            new_generation[i] = mutate_invert(individual)  # Mutate and update the individual in the dictionary


def create_generation(sorted_gen):
    new_gen = {}

    elite_end = int(elitism * number_of_individuals)  # elitism

    if elite_end != 0:
        for j in range(elite_end):
            new_gen[j] = sorted_gen[j]  # Keep elite individuals

    start_fresh = int(elitism * number_of_individuals)  # fresh individuals
    end_fresh = int((elitism + new_individuals) * number_of_individuals)
    for k in range(start_fresh, end_fresh):
        new_gen[k] = spawn_individual()  # Create new individuals

    roulette_selection(sorted_gen, new_gen)

    mutation(new_gen)  # Apply the "invert" mutation

    # Convert the dictionary into a list of Individual objects
    new_gen = list(new_gen.values())

    return new_gen



if __name__ == "__main__":
    # Initialize the population
    spawn_generation()
    best_individual = None
    sorted_individuals = {}

    for generation in range(number_of_gens):
        # Calculate fitness for each individual in the population
        for i in range(number_of_individuals-1):
            virtual_machine(individuals[i])
            found_treasures(individuals[i])
            individuals[i].fitness_func = float(calculate_fitness(individuals[i]))

        # Sort the population based on fitness in descending order
        sorted_individuals = sorted([ind for ind in individuals if isinstance(ind, Individual)],
                                    key=lambda ind: ind.fitness_func, reverse=True)

        # Keep track of the best individual


        # Create a new generation
        new_generation = create_generation(sorted_individuals)

        # Update the population with the new generation
        individuals = new_generation  # Just replace the list with the new one
        sorted_individuals = {}


    print("Best Individual - FINAL:", individuals[0].fitness_func)
    print("Path:", individuals[0].path)

    # Print the best individual's path and fitness
