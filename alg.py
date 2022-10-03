from math import dist, sin, cos, pi
import random

def normal_distribution(ratio,n):
    pesos =[]
    res = []
    lambda1 = 0
    lambda2 = 1
    pesos.append(lambda1)
    pesos.append(lambda2)
    res.append(pesos)
    for i in range(0,n-1):
        p = []
        lambda1 = lambda1+ratio
        lambda2 = lambda2-ratio
        p.append(round(lambda1,4))
        p.append(round(lambda2,4))
        res.append(p)

    return res

def distance(pesos):
    "Uso de la distancia euclídea:"
    '''Esta función calcula la distancia euclídea para cada par de vectores de pesos consecutivos
    Devuelve un diccionario (clave=subproblema(vector), valor = distancia)'''
    res = []
    i = 0
    for peso in pesos:
        j = 0
        distances_i = []
        for subpeso in pesos:
            
            distance = dist(peso,subpeso)
            distances_i.append(distance)
            j = j+1
        res.append(distances_i)
        i = i+1
    return res

#obtener indice de una lista
def get_index(lista,valor):
    for i in range(0,len(lista)):
        if lista[i] == valor:
            return i
    return -1

# para cada subproblema identificar los los T vectores mas cercanos
def t_nearest(distancias,t):
    "Cálculo de los T vectores más cercanos"
    res = []
    for subproblema in distancias:
        vectores = []
        for i in range(0,t):
            minimo = min(subproblema)
            indice = get_index(subproblema,minimo)
            vectores.append(indice)
            subproblema[indice] = 1000000
        res.append(vectores)
    return res


# Generar una poblacion de individuos aleatoriamente
def generate_population(pob, xli, xui):   
    population = []
    for i in range(pob):
        individual = []
        for j in range(len(xli)):
            individual.append(random.uniform(xli[j], xui[j]))
        population.append(individual)
    return population

# Evaluar la población con la función zdt3 (objetivo)
def evaluate_population(population):
    evaluated_population = []
    for individual in population:
        evaluated_population.append(zdt3(individual))
    return evaluated_population

# zdt3 function implementation
def zdt3(x):    
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    f2 = g * (1 - (x[0] / g) ** 0.5 - x[0] / g * sin(10 * pi * x[0]))
    return [f1, f2] 

def find_best(fitness):
    mejores = []
    minimo_f1 = fitness[0][0]
    minimo_f2 = fitness[0][1] 

    "Búsqueda de los menores valores de f1 y f2"
    for i in range(0,len(fitness)):
        if fitness[i][0] < minimo_f1:
            minimo_f1 = fitness[i][0]   
        if fitness[i][1] < minimo_f2:
            minimo_f2 = fitness[i][1]
    
    mejores.append([minimo_f1,minimo_f2])

def init(pob,xli,xui):

    "El número de subproblemas es igual al tamaño de población"
    subproblemas = pob

    "1 PASO: DISTRIBUCIÓN UNIFORME DE LOS N VECTORES PESO"
    '''ratio por el que vamos a aumentar y decrementar 
    el peso de cada uno de los subproblemas'''
    ratio = 1/(subproblemas-1)
    pesos = normal_distribution(ratio,subproblemas)

    "2 PASO: CÁLCULO DE LA DISTANCIA EUCLÍDEA"
    distancias =  distance(pesos)
    
    "3 PASO: IDENTIFICACIÓN DE LOS T VECTORES PESOS"

    "Hacemos una copia de las distancias"
    distancias_aux = distancias.copy()
    t = 0.2*pob
    selector_cercanos = t_nearest(distancias_aux,t)
    
    "4 PASO: GENERACIÓN DE LA POBLACIÓN ALEATORIA"
    poblacion = generate_population(pob,xli,xui)

    "5 PASO: EVALUACIÓN DE LA POBLACIÓN"
    fitness = evaluate_population(poblacion)
    "Seleccionamos los mejores valores objetivos fi encontrados"
    z = find_best(fitness)   

