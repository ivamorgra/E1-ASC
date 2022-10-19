from cProfile import label
from math import dist, sin, cos, pi
import random
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import math

def show_initial_graph(f1best,f2best,f1pa,f2pa,f1,f2):
    plt.scatter(f1best,f2best, color='yellow',label='Best')
    plt.scatter(f1pa,f2pa,color='red',label='Pareto Front')
    plt.scatter(f1,f2, color='blue',label='Solution')
    plt.title(' ZDT3 (Población inicial) ')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.legend()
    plt.show()

def normal_distribution(ratio,pob):
    pesos =[]
    res = []
    lambda1 = 0
    lambda2 = 1
    pesos.append(lambda1)
    pesos.append(lambda2)
    res.append(pesos)
    for i in range(0,pob-1):
        p = []
        lambda1 = lambda1+ratio
        lambda2 = abs(lambda2-ratio)
        p.append(round(lambda1,4))
        p.append(round(lambda2,4))
        res.append(p)

    return res

def distance(pesos):
    "Uso de la distancia euclídea:"
    
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
    f1 = []
    f2 = []
    for i in range(0,len(population)):
        solucion = zdt3(population[i])
        evaluated_population.append(solucion)
        f1.append(solucion[0])
        f2.append(solucion[1])
    return f1,f2,evaluated_population

# Evaluar la población con la función zdt3 (objetivo)
def evaluate_individual(individual):
    evaluated_population = []
    solucion = zdt3(individual)
    evaluated_population.append(solucion)
    return solucion[0],solucion[1],evaluated_population

# zdt3 function implementation
def zdt3(x):  

    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    f2 = g * (1 - (x[0] / g) ** 0.5 - x[0] / g * sin(10 * pi * x[0]))

    return [f1, f2] 

def find_best(f):
    return min(f)



def tchebycheff(x,pesos,z):

    ''' Para la actualización de los vecinos debemos tener en cuenta la formulación de Tchebycheff
    que es la siguiente:
    f(x) = max(wi* abs(fi(x)-zi)))
    donde wi es el peso del subproblema i y fi(x) es la función objetivo del subproblema i
    '''
    res = []
    for i in range(0,len(pesos)):
        functions = zdt3(x)
        value = pesos[i]* abs(functions[i]-z[i])
        res.append(value)
    return max(res)

def update_neihbours(index,poblacion,solucion,lista,pesos,z):
    ''' '''
    
    for vecino in lista[index]:
        if tchebycheff(solucion,pesos[vecino],z) <= tchebycheff(poblacion[vecino],pesos[vecino],z):
            poblacion[vecino] = solucion
    return poblacion


def gaussian(individual,xli,xui):
    max_ = max(individual)
    min_ = min(individual)
    pr = 1/30
    aux = 0.0
    desviation = (max_-min_)/20

    for i in range(0,len(individual)):
        if random.random() < pr:
            aux = individual[i] + random.gauss(0,desviation)
            if aux > xui[i]:
                aux = xui[i]
                "Miramos el limite inferior"
            if aux < xli[i]:
                aux = xli[i]
            individual[i] = aux
    return individual

def operator(lista,poblacion,f,xui,xli,cr,z,pesos):

    ''' Como parámetro debe ser una lista de listas
     - Parámetro f: factor de cruce
     - index_ind es el índice del individuo que se va a mutar
     - xui: Parámetro que indica el límite superior del espacio de búsqueda
     - xli: Parámetro que indica el límite inferior del espacio de búsqueda'''
    
    functions_one = []
    functions_two = []
    
    for j in range(0,len(poblacion)):
        vecinos_copy = lista[j].copy()
        n = []
        ind = []
        for i in range(0,3):
            index_v = random.choice(vecinos_copy)
                
            n.append(index_v)
            vecinos_copy.remove(index_v)
        
        ''' APLICAMOS MUTACIÓN'''
        "Por cada dimensión (son 30)"
        for i in range(0,30):
            r1 =  poblacion[n[0]]
            r2 = poblacion[n[1]]
            r3 = poblacion[n[2]]
            
            individual = r1[i] + f * (r2[i] - r3[i])
            
            "Antes de añadir a lista nos aseguramos de que el individuo no sobrepase los límites del espacio de búsqueda"        "Miramos el limite superior"
            if individual > xui[i]:
                individual = xui[i]
                "Miramos el limite inferior"
            if individual < xli[i]:
                individual = xli[i]


            ind.append(individual)
            ''' APLICAMOS CRUCE (factor de cruce CR)
            Para ello nos vamos recorriendo los índices del individuo tras aplicar mutación'''

        for i in range(0,len(ind)):
            if random.random() < cr:
                ind[i] = poblacion[j][i]   

        ind = gaussian(ind,xli,xui)
        "Actualización de z"
        '''PASO 2: EVALUACIÓN DE LA NUEVA POBLACIÓN'''
        f1,f2,fitness = evaluate_individual(ind)
        functions_one.append(f1)
        functions_two.append(f2)
        '''PASO 3: ACTUALIZACIÓN DE Z'''
        if f1 < z[0]:
            z[0] = f1
        if f2 < z[1]:
            z[1] = f2
        
        '''PASO 4: ACTUALIZACIÓN DE LOS VECINOS'''
        poblacion = update_neihbours(j,poblacion,ind,lista,pesos,z)
    
    return poblacion,z,functions_one,functions_two

def pareto_front():

    f = open("PF.dat")
    f1 = []
    f2 = []
    for line in f:
        functions = line.split("\t")
        f1.append(float(functions[0]))
        f2.append(float(functions[1]))
    f.close()
    
    return f1,f2

def init(pob,xli,xui,unique,seed):

    if unique==False:
        random.seed(seed)

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
    selector_cercanos = t_nearest(distancias_aux,round(t))
    
    "4 PASO: GENERACIÓN DE LA POBLACIÓN ALEATORIA"
    poblacion = generate_population(pob,xli,xui)

    "5 PASO: EVALUACIÓN DE LA POBLACIÓN"
    f1,f2,fitness = evaluate_population(poblacion)
    "Seleccionamos los mejores valores objetivos fi encontrados"
    f1best = find_best(f1)
    f2best = find_best(f2)
    z = [f1best,f2best] 

    "Representación gráfica"
    if unique==True:
        f1pa,f2pa = pareto_front()
        show_initial_graph(f1best,f2best,f1pa,f2pa,f1,f2)

    return poblacion,z, selector_cercanos,pesos



def iterative(poblacion,xli,xui,selector_cercanos,pesos,f,z):
    ''' MÉTODO ITERATIVO DEL ALGORITMO BASADO EN AGREGACIÓN'''

    '''PASO 1: SELECCIÓN ALEATORIA DE LOS ÍNDICES VECINOS DE CADA SUBPROBLEMA GENERAR UNA SOLUCIÓN CON OPERADORES
    EVOLUTIVOS'''
    "Nueva población de individuos"
    cr = 0.5
        
    poblacion,z,f1,f2 = operator(selector_cercanos,poblacion,f,xui,xli,cr,z,pesos)
            

        

    return poblacion,z,f1,f2

