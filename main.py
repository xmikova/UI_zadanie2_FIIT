import random
import numpy

mem_cells_count = 64
first_gen_mem_cells_count = 32
max_steps = 500
elitism = 0.01  # percent of new individuals made by elitism
new_individuals = 0.15  # percent of new individuals made for new generation
mutation = 0.1  # probability of mutation
tournament = 0.2  # number of individuals in tournament

number_of_gens = 300
number_of_individuals = 80

#TODO : implementovat parser pre txt vstupu

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


def spawn_generation():
    individuals = {}

    for i in range(number_of_individuals):
        individuals[i] = Individual([], [], set(), 0)
        for j in range(mem_cells_count):
            if j <= first_gen_mem_cells_count:
                individuals[i].memory_cells.append(numpy.uint8(random.randrange(256)))
            else:
                individuals[i].memory_cells.append(numpy.uint8(0))


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





if __name__ == "__main__":
    print("hello")