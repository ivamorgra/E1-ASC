from math import dist, sin, cos, pi
import random
from turtle import color
import matplotlib.pyplot as plt

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
    f1 = []
    f2 = []
    for individual in population:
        solucion = zdt3(individual)
        evaluated_population.append(solucion)
        f1.append(solucion[0])
        f2.append(solucion[1])
    return f1,f2,evaluated_population

# zdt3 function implementation
def zdt3(x):  

    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    f1 = x[0]
    f2 = g * (1 - (x[0] / g) ** 0.5 - x[0] / g * sin(10 * pi * x[0]))

    return [f1, f2] 

def find_best(f):

    order_list = sorted(f)

    return order_list[0]
    
def graph(f1,f2):
    plt.ylim([0,7])
    plt.scatter(f1,f2)
    

#plot pareto front
def pareto(f):  
    x = [i / 100 for i in range(100)]
    y = [1 - (i / 100) ** 0.5 for i in x]
    plt.plot(x, y, color='black')

# Select random index from list
def select_random_index(lista,poblacion,f,xui,xli):
    ''' Como parámetro debe ser una lista de listas
     - Parámetro f: factor de cruce
     - xui: Parámetro que indica el límite superior del espacio de búsqueda
     - xli: Parámetro que indica el límite inferior del espacio de búsqueda'''
    res = []
    for vecinos in lista:
        vecinos_copy = vecinos.copy()
        n = []
        ind = []
        for i in range(0,3):
            index_v = random.choice(vecinos_copy)
            n.append(index_v)
            vecinos_copy.remove(index_v)
        
        "Por cada dimensión (son 30)"
        for i in range(0,30):
            
            individual = poblacion[n[0]][i] + f * (poblacion[n[1]][i] - poblacion[n[2]][i])

            "Antes de añadir a lista nos aseguramos de que el individuo no sobrepase los límites del espacio de búsqueda"
            "Miramos el limite superior"
            if individual > xui[i]:
                individual = xui[i]
            "Miramos el limite inferior"
            if individual < xli[i]:
                individual = xli[i]

            ind.append(individual)
        res.append(ind)
    return res


def cruce(p1,p2):
    '''Esta función aplica el operador de cruce a dos vectores de pesos'''
    res = []
    for i in range(0,len(p1)):
        res.append((p1[i]+p2[i])/2)
    return res

def operadores(poblacion,selection,prob_cruce,prob_mutacion):
    '''Esta función aplica el operador de cruce a los individuos seleccionados'''
    
    
    return res

    

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
    plt.scatter(f1best,f2best, color='red')
    graph(f1,f2)
    plt.show()
    return poblacion,z, selector_cercanos


    def iterative(pob,xli,xui,f):
        poblacion,z, selector_cercanos = init(pob,xli,xui)

        ''' MÉTODO ITERATIVO DEL ALGORITMO BASADO EN AGREGACIÓN'''

        '''PASO 1: SELECCIÓN ALEATORIA DE LOS ÍNDICES VECINOS DE CADA SUBPROBLEMA GENERAR UNA SOLUCIÓN CON OPERADORES
        EVOLUTIVOS'''
        selection = select_random_index(selector_cercanos,poblacion,f,xui,xli)
        
        

        
        pass

