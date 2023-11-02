import random
import numpy


def get_user_input(prompt, default_value):
    user_input = input(f"{prompt} (default hodnota: {default_value}): ")
    if user_input == "":
        return default_value
    return type(default_value)(user_input)


print()
print("*----------------------------------------------------------------------------*")
print("|                 Hľadanie pokladu - evolučný algoritmus                     |")
print("|                           Autor: Petra Miková                              |")
print("|                             UI, ZS 2023/2024                               |")
print("*----------------------------------------------------------------------------*")
print()
print("Nasleduje input hodnôt pre algoritmus, ak chcete ponechať defaultnu hodnotu, stlačte enter:")
# Defaultne hodnoty, s ktorými sa program spúšťa ak ich používateľ nechce zadať sám
mem_cells_count = get_user_input("Zadajte počet pamäťových buniek pre virtuálny stroj", 64)
first_gen_mem_cells_count = get_user_input("Zadajte počet pamäťových buniek ktoré sa inicializujú v prvej generácii", 32)
max_steps = get_user_input("Zadajte počet krokov, po koľkých sa zastaví hľadanie", 500)
elitism = get_user_input("Zadajte percento nových jedincov vytvorených elitarizmom", 0.02)
new_individuals = get_user_input("Zadajte percento nových jedincov vytvorených v novej generácii", 0.15)
mutation_rate = get_user_input("Zadajte pravdepodobnosť mutácie", 0.05)
tournament = get_user_input("Zadajte percento jedincov v turnaji", 0.2)
number_of_gens = get_user_input("Zadajte počet generácií", 200)
number_of_individuals = get_user_input("Zadajte počet jedincov v jednej generácii", 80)

# Trieda, ktorá obsahuje štruktúru jedinca - jeho prejdenú cestu, pamäťové bunky, nájdené poklady, a fitness funkciu
class Individual:
    def __init__(self, path, memory_cells, treasures_found, fitness_func):
        self.path = path
        self.memory_cells = memory_cells
        self.treasures_found = treasures_found
        self.fitness_func = fitness_func


# Premenné týkajuce sa mriežky, pozícii pokladov a finálnych výsledkov algoritmu
map_size = (7, 7)
all_treasures = {(1, 4), (2, 2), (3, 6), (4, 1), (5, 4)}
start_position = (6, 3)
individuals = []
best_individual = None


# Funkcia, ktorá vytvorí prvotnú generáciu a nainicializuje ju jedincami tak, aby vedela evolúcia ďalej konvergovať
def spawn_generation():
    for i in range(number_of_individuals):
        individuals.append(spawn_individual())
        individuals[i].memory_cells = []
        for j in range(mem_cells_count):
            if j <= first_gen_mem_cells_count:
                individuals[i].memory_cells.append(numpy.uint8(random.randrange(256)))
            else:
                individuals[i].memory_cells.append(numpy.uint8(0))


# Funkcia, ktorá vytvorí nového jedinca do danej generácie
def spawn_individual():
    path = []
    memory_cells = [numpy.uint8(random.randint(0, 255)) for _ in range(mem_cells_count)]
    treasures_found = set()
    fitness_func = 0

    new_individual = Individual(path, memory_cells, treasures_found, fitness_func)

    return new_individual


# Jednoduchá funkcia, ktorá vráti znakovú podobu pohybu pre výpis cesty jedinca
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


# Funkcia, ktorá simuluje virtuálny stroj pre každého jedinca, za inštrukcie berieme pamäťové bunky jedinca
# ktoré si rozsparsujeme na 8 bitové binárne stringy a podľa definície virtuálneho stroja vykonávame dané operácie, vykonávajú
# sa až po maximálny počet krokov
def virtual_machine(individual):
    steps_count = 0
    cells_count = 0
    instructions = individual.memory_cells

    while steps_count <= max_steps and 0 <= cells_count < mem_cells_count:
        operation = format(instructions[cells_count], '08b')[0:2]
        address = int(format(instructions[cells_count], '08b')[2:], 2)

        if operation == "00":  # Inkrementácia
            individual.memory_cells[address] += 1
            individual.memory_cells[address] = numpy.uint8(individual.memory_cells[address])

        elif operation == "01":  # Dekrementácia
            individual.memory_cells[address] -= 1
            individual.memory_cells[address] = numpy.uint8(individual.memory_cells[address])

        elif operation == "10":  # Skok
            cells_count = address

        elif operation == "11":  # Zápis cesty
            move = format(instructions[cells_count], '08b')[6:]
            individual.path.append(return_move(move))

        steps_count += 1
        cells_count += 1


# Samotné prechádzanie mriežkou a hľadanie pokladov, ukončujeme ak jedinec našiel všetky poklady alebo sa posunul mimo mriežky
def treasure_hunt(individual):
    row, column = start_position
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
            break

        current_position = (row, column)
        if current_position in all_treasures and current_position not in individual.treasures_found:
            individual.treasures_found.add(current_position)

        updated_path.append(move)

    individual.path = updated_path

    if len(individual.treasures_found) == len(all_treasures):
        return


# Výpočet fitnes funkcie jedinca, berieme do úvahy najmä počet nájdených pokladov a taktiež penalizujeme dĺžku cesty,
# čím vyššia fitness funkcia tým je jedinec lepší
def calculate_fitness(individual):
    if not individual.path:
        return 0

    treasures_found = len(individual.treasures_found)
    total_steps = len(individual.path)

    treasures_weight = 10
    steps_weight = -0.001

    fitness = (treasures_weight * treasures_found) + (steps_weight * total_steps)

    return fitness


# Vykonanie kríženia spôsobom že sa náhodne vyberie index kde bude dochádzať ku kríženiu a po tento index
# sa novému jedincovi pridajú bunky prvého rodiča, po indexe toho druhého
def crossover(parent1, parent2):
    crossover_point = random.randint(1, mem_cells_count - 1)
    child_memory_cells = parent1.memory_cells[:crossover_point] + parent2.memory_cells[crossover_point:]
    return child_memory_cells


# Selekcia - typ ruleta - vyberajú sa za rodičov jedinci spôosobom, že čím vyššiu majú fitness funkciu,
# tým majú vyššiu šancu vybratia do kríženia
def roulette_selection(sorted_gen, new_generation):
    start_index = int((elitism + new_individuals) * number_of_individuals)
    weights = [ind.fitness_func for ind in sorted_gen]

    for i in range(start_index, number_of_individuals):
        parent1 = random.choices(sorted_gen, weights=weights)[0]
        parent2 = random.choices(sorted_gen, weights=weights)[0]

        # Ensure that parent1 and parent2 are different individuals
        while parent1 == parent2:
            parent2 = random.choices(sorted_gen, weights=weights)[0]

        # Create a new offspring Individual object and perform crossover on memory_cells
        offspring_memory_cells = crossover(parent1, parent2)
        offspring = Individual([], offspring_memory_cells, set(), 0)
        new_generation[i] = offspring


# Selekcia - typ turnaj - náhodne sa vyberie pevný počet jedincov z populácie na základe percenta turnaja a najvhodnejší jedinec z turnaja
# je vybraný ako rodič pre nasledujúcu generáciu - toto sa opakuje pre výber viacerých rodičov na generovanie ďalšej generácie jedincov
def tournament_selection(sorted_gen, new_generation):
    start_index = int((elitism + new_individuals) * number_of_individuals)
    num_tournament = int(tournament * number_of_individuals)
    for i in range(start_index, number_of_individuals):
        parent1 = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x.fitness_func)[0]
        parent2 = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x.fitness_func)[0]

        while numpy.array_equal(parent2.memory_cells, parent1.memory_cells):
            parent2 = sorted(random.choices(sorted_gen, k=num_tournament), reverse=True, key=lambda x: x.fitness_func)[0]

        offspring_memory_cells = crossover(parent1, parent2)
        offspring = Individual([], offspring_memory_cells, set(), 0)
        new_generation[i] = offspring


# Invertovanie pre vykonávanie mutácie, pre každú bunku pamäte v jedincovi sa náhodne rozhodne, či sa vykoná mutácia,
# a ak áno, vyberie sa náhodné miesto (bit) v danej bunke pamäte a zmení sa jeho hodnota
def invert_cells(individual):
    mutated_memory_cells = list(individual.memory_cells)

    for index, value in enumerate(individual.memory_cells):
        if random.random() < mutation_rate:
            shift = random.randint(0, 7)
            mutated_memory_cells[index] = numpy.uint8(value ^ (1 << shift))

    mutated_individual = Individual(individual.path, mutated_memory_cells, individual.treasures_found, individual.fitness_func)
    return mutated_individual


#Vykoná mutáciu pre celú generáciu jedincov
def mutation(new_generation):
    start_index = int((elitism + new_individuals) * number_of_individuals)
    for i in range(start_index, number_of_individuals):
        if i in new_generation:
            individual = new_generation[i]
            new_generation[i] = invert_cells(individual)


# Základná funckia pre vytvorenie každej ďalšej generácie, použije sa elitarizmus pre zachovanie elitných jedincov, vykoná sa na základe výberu
# druh selekcie, zmutuje sa generácia a vráti túto novú generáciu
def create_generation(sorted_gen, selection):
    new_gen = {}

    elite_end = int(elitism * number_of_individuals)

    if elite_end != 0:
        for j in range(elite_end):
            new_gen[j] = sorted_gen[j]

    start_fresh = int(elitism * number_of_individuals)
    end_fresh = int((elitism + new_individuals) * number_of_individuals)
    for k in range(start_fresh, end_fresh):
        new_gen[k] = spawn_individual()

    if selection == "ruleta":
        roulette_selection(sorted_gen, new_gen)
    else:
        tournament_selection(sorted_gen, new_gen)

    mutation(new_gen)
    new_gen = list(new_gen.values())

    return new_gen


selection_choice = input("Zadajte metódu selekcie (ruleta(default) / turnaj) (pre ponechanie default stlačte enter): ")
if selection_choice != "ruleta" or selection_choice != "turnaj":
    print("Používa sa defaultna metóda - ruleta")
    selection_choice = "ruleta"


#Spustenie programu a nájdenie najlepšieho hľadača pokladov
if __name__ == "__main__":
    continue_generating = True
    counter = 1
    sorted_individuals = {}

    while continue_generating:
        spawn_generation()

        for generation in range(number_of_gens):
            for i in range(number_of_individuals-1):
                virtual_machine(individuals[i])
                treasure_hunt(individuals[i])
                individuals[i].fitness_func = float(calculate_fitness(individuals[i]))

            sorted_individuals = sorted([ind for ind in individuals if isinstance(ind, Individual)],
                                        key=lambda ind: ind.fitness_func, reverse=True)

            new_generation = create_generation(sorted_individuals, selection_choice)

            individuals = new_generation

            if best_individual is None or (
                    len(sorted_individuals) != 0 and (sorted_individuals[0].fitness_func > best_individual.fitness_func)):
                best_individual = sorted_individuals[0]

            sorted_individuals = {}

            print("Najlepší jedinec pre", counter, ". generáciu:")
            print("Fitnes funkcia: ", individuals[0].fitness_func)
            print("Cesta:", individuals[0].path)

            counter += 1

        # Pýtame sa používateľa či chce pokračovať alebo nie
        user_input = input("Chcete pokračovať v generovaní? (ano/nie): ")
        if user_input.lower() != "ano":
            continue_generating = False
            print()
            print("Najlepší jedinec celkovo: ")
            print("Fitnes funkcia: ", best_individual.fitness_func)
            print("Cesta:", best_individual.path)
