import numpy as np
from main import TrafficLightClassification
from load_data import load_dataset
from config import *
from standardize import standardize_image
import tqdm

def random_hue():
    lower_limit = np.random.randint(0,181)
    upper_limit = np.random.randint(lower_limit,181)
    return lower_limit, upper_limit

def random_sv():
    lower_saturation_limit_ratio = round(0.5+np.random.rand(),3)
    lower_value_limit = np.random.randint(0,255)
    return lower_saturation_limit_ratio, lower_value_limit

def random_chromosome():
    return [*random_sv(), *random_hue(), *random_hue(), *random_hue()]

def random_population(population_size):
    return [random_chromosome() for _ in range(population_size)]

def crossover(offspring):
    new_gens = []
    for i in range(len(offspring)):
        value = offspring[i][1][i]
        for j in range(len(offspring)):
            if i == j:
                continue
            new_gen = offspring[j][1].copy()
            new_gen[i] = value
            new_gens.append(new_gen)
    
    return new_gens

def crossover2(offspring):
    new_gens = []
    for i in range(len(offspring)-1):
        for j in range(i+1, len(offspring)):
            new_gen = (np.array(offspring[i][1])+np.array(offspring[j][1]))/2
            new_gens.append(new_gen.astype(np.int16).tolist())
    return new_gens

def mutate(population):
    new_gens = []
    for chromosome in population:
        rand_mult = (np.random.rand()/5+0.9)
        chromosome = np.array(chromosome)*rand_mult
        new_gens.append([round(chromosome[0],3)] + chromosome[1:].astype(np.int16).tolist())
    return new_gens

def fitness(classifier, dataset, chromosome):
    return classifier.classify(dataset, chromosome)

def optimize(classifier, dataset, population_size, max_generations, min_fittness):
    fittest_solution = ((0,0,0), [])
    population = random_population(population_size)
    offspring = []
        
    for i in tqdm.tqdm(range(max_generations), total=max_generations):
        # evaluate current population
        for j, chromosome in enumerate((pbar := (tqdm.tqdm(population, total=len(population))))):
            pbar.set_description(f"Current Accuracy {round(fittest_solution[0][-1],4)}")
            correct, missed, accuracy = fitness(classifier, dataset, chromosome)
            if accuracy > fittest_solution[0][-1]:
                fittest_solution = ((correct, missed, accuracy), chromosome)
            population[j] = (accuracy, chromosome)
        population.sort(key= lambda c: c[0], reverse=True)
        offspring = population[:4] + [population[i] for i in np.random.default_rng().choice(len(population)-4, size=4, replace=False)+4]
        population = crossover(offspring)
        population += crossover2(offspring)
        population += mutate(population)
        population += random_population(population_size-len(population))
        print(fittest_solution)

    print(fittest_solution)

if __name__ == '__main__':
    TRAIN_IMAGE_LIST = load_dataset(IMAGE_DIR_TRAINING)
    standardized_train_list = standardize_image(TRAIN_IMAGE_LIST, *STD_IMAGE_SIZE)
    TEST_IMAGE_LIST = load_dataset(IMAGE_DIR_TEST)
    standardized_test_list = standardize_image(TEST_IMAGE_LIST, *STD_IMAGE_SIZE)

    classifier = TrafficLightClassification()
    
    optimize(classifier, standardized_train_list, 1000, int(100), min_fittness=0.99)