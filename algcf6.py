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

# Evaluar la población con la función cf6 (objetivo)
def evaluate_population4d(population,peso):
    evaluated_population = []
    f1 = []
    f2 = []
    pens = []
    for i in range(0,len(population)):
        solucion,pen = cf6(population[i],4,peso)
        evaluated_population.append(solucion)
        f1.append(solucion[0])
        f2.append(solucion[1])
        pens.append(pen)
    evaluated_population.append(solucion)
    return f1,f2,evaluated_population,pens

# Evaluar la población con la función cf6 (objetivo)
def evaluate_population16d(population,peso):
    evaluated_population = []
    f1 = []
    f2 = []
    pens = []
    for i in range(0,len(population)):
        solucion,pen = cf6(population[i],16,peso)
        evaluated_population.append(solucion)
        f1.append(solucion[0])
        f2.append(solucion[1])
        pens.append(pen)
    evaluated_population.append(solucion)
    return f1,f2,evaluated_population,pens

# Evaluar el individuo con la función cf6 (objetivo)
def evaluate_individual4d(individual,peso):
    evaluated_population = []
    solucion,pen = cf6(individual,4,peso)
    evaluated_population.append(solucion)
    return solucion[0],solucion[1],evaluated_population,pen

# Evaluar el individuo con la función cf6 (objetivo)
def evaluate_individual16d(individual,peso):
    evaluated_population = []
    solucion,pen = cf6(individual,16,peso)
    evaluated_population.append(solucion)
    return solucion[0],solucion[1],evaluated_population,pen

#CF6 function implementation
def cf6(x,n,peso):

    pen,pen2 = restrictions(x,n,peso)

    yj1 = 0
    for i in range(1,n+1,2):
        if i%2 == 1:
            yj1 = yj1 + (x[i-1] - 0.8*x[0]*math.cos(6*math.pi*x[0]+i*math.pi/n))**2
    
    yj2 = 0
    for i in range(1,n+1,2):
        if i%2 == 0:
            yj2 = yj2 + (x[i-1] - 0.8*x[0]*math.sin(6*math.pi*x[0]+i*math.pi/n))**2
       
    f1 = x[0] + yj1
    f2 = (1-x[0])**2 + yj2
    
    return [f1 + pen, f2 + pen],pen2

def find_best(f):
    return min(f)


def tchebycheff4d(x,pesos,z,peso):

    ''' Para la actualización de los vecinos debemos tener en cuenta la formulación de Tchebycheff
    que es la siguiente:
    f(x) = max(wi* abs(fi(x)-zi)))
    donde wi es el peso del subproblema i y fi(x) es la función objetivo del subproblema i
    '''
    res = []
    for i in range(0,len(pesos)):
        functions,pen = cf6(x,4,peso)
        value = pesos[i]* abs(functions[i]-z[i])
        res.append(value)
    return max(res)

def tchebycheff16d(x,pesos,z,peso):

    ''' Para la actualización de los vecinos debemos tener en cuenta la formulación de Tchebycheff
    que es la siguiente:
    f(x) = max(wi* abs(fi(x)-zi)))
    donde wi es el peso del subproblema i y fi(x) es la función objetivo del subproblema i
    '''
    res = []
    for i in range(0,len(pesos)):
        functions,pen = cf6(x,16,peso)
        value = pesos[i]* abs(functions[i]-z[i])
        res.append(value)
    return max(res)

def update_neihbours4d(index,poblacion,solucion,lista,pesos,z,peso):
    ''' '''
    
    for vecino in lista[index]:
        if tchebycheff4d(solucion,pesos[vecino],z,peso) <= tchebycheff4d(poblacion[vecino],pesos[vecino],z,peso):
            poblacion[vecino] = solucion
    return poblacion

def update_neihbours16d(index,poblacion,solucion,lista,pesos,z,peso):
    ''' '''
    
    for vecino in lista[index]:
        if tchebycheff16d(solucion,pesos[vecino],z,peso) <= tchebycheff16d(poblacion[vecino],pesos[vecino],z,peso):
            poblacion[vecino] = solucion
    return poblacion


def gaussian4d(individual,xli,xui):
    max_ = max(individual)
    min_ = min(individual)
    pr = 1/4
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

def gaussian16d(individual,xli,xui):
    max_ = max(individual)
    min_ = min(individual)
    pr = 1/16
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

def restrictions(individual,n,peso):
    ''' Parámetros:
    - individual: es el individuo que se va a evaluar
    - n: es el número de dimensiones del problema
    - peso: es el peso que se le va a asignar a la penalización'''
    pen = 0.0

    r1 = individual[1]-0.8*individual[0]*math.sin(6*math.pi*individual[0]+2*math.pi/n)+np.sign(0.5*(1-individual[0])-(1-individual[0])**2)*math.sqrt(abs(0.5*(1-individual[0])-(1-individual[0])**2))
    r2 = individual[3]-0.8*individual[0]*math.sin(6*math.pi*individual[0]+4*math.pi/n)+np.sign(0.25*math.sqrt(1-individual[0])-0.5*(1-individual[0]))*math.sqrt(abs(0.25*math.sqrt(1-individual[0])-0.5*(1-individual[0])))
    if r1< 0:
        pen = pen + (r1-0)**2
    if r2 < 0:
        pen = pen + (r2-0)**2
    
    return pen*peso, pen


def operator4d(lista,poblacion,f,xui,xli,cr,z,pesos,peso):

    ''' Como parámetro debe ser una lista de listas
     - Parámetro f: factor de cruce
     - index_ind es el índice del individuo que se va a mutar
     - xui: Parámetro que indica el límite superior del espacio de búsqueda
     - xli: Parámetro que indica el límite inferior del espacio de búsqueda'''
    
    functions_one = []
    functions_two = []
    pens = []
    for j in range(0,len(poblacion)):
        vecinos_copy = lista[j].copy()
        n = []
        ind = []
        for i in range(0,3):
            index_v = random.choice(vecinos_copy)
                
            n.append(index_v)
            vecinos_copy.remove(index_v)
        
        ''' APLICAMOS MUTACIÓN'''
        "Por cada dimensión (son 16)"
        for i in range(0,4):
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

        ind = gaussian4d(ind,xli,xui)
        "Actualización de z"
        '''PASO 2: EVALUACIÓN DE LA NUEVA POBLACIÓN'''
        f1,f2,fitness,pen = evaluate_individual4d(ind,peso)
        functions_one.append(f1)
        functions_two.append(f2)
        pens.append(pen)
        '''PASO 3: ACTUALIZACIÓN DE Z'''
        if f1 < z[0]:
            z[0] = f1
        if f2 < z[1]:
            z[1] = f2
        
        '''PASO 4: ACTUALIZACIÓN DE LOS VECINOS'''
        poblacion = update_neihbours4d(j,poblacion,ind,lista,pesos,z,peso)
    
    return poblacion,z,functions_one,functions_two,pen


def operator16d(lista,poblacion,f,xui,xli,cr,z,pesos,peso):

    ''' Como parámetro debe ser una lista de listas
     - Parámetro f: factor de cruce
     - index_ind es el índice del individuo que se va a mutar
     - xui: Parámetro que indica el límite superior del espacio de búsqueda
     - xli: Parámetro que indica el límite inferior del espacio de búsqueda'''
    
    functions_one = []
    functions_two = []
    pens = []
    for j in range(0,len(poblacion)):
        vecinos_copy = lista[j].copy()
        n = []
        ind = []
        for i in range(0,3):
            index_v = random.choice(vecinos_copy)
                
            n.append(index_v)
            vecinos_copy.remove(index_v)
        
        ''' APLICAMOS MUTACIÓN'''
        "Por cada dimensión (son 16)"
        for i in range(0,16):
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

        ind = gaussian16d(ind,xli,xui)
        "Actualización de z"
        '''PASO 2: EVALUACIÓN DE LA NUEVA POBLACIÓN'''
        f1,f2,fitness,pen = evaluate_individual16d(ind,peso)
        functions_one.append(f1)
        functions_two.append(f2)
        pens.append(pen)
        '''PASO 3: ACTUALIZACIÓN DE Z'''
        if f1 < z[0]:
            z[0] = f1
        if f2 < z[1]:
            z[1] = f2
        
        '''PASO 4: ACTUALIZACIÓN DE LOS VECINOS'''
        poblacion = update_neihbours16d(j,poblacion,ind,lista,pesos,z,peso)
    
    return poblacion,z,functions_one,functions_two,pens

def pareto_front():

    f = open("PFCF6.dat")
    f1 = []
    f2 = []
    for line in f:
        functions = line.split("\t")
        f1.append(float(functions[0]))
        f2.append(float(functions[1]))
    f.close()
    
    return f1,f2

def init4d(pob,xli,xui,peso,unique,seed):


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
    f1,f2,fitness,pens = evaluate_population4d(poblacion,peso)
    "Seleccionamos los mejores valores objetivos fi encontrados"
    f1best = find_best(f1)
    f2best = find_best(f2)
    z = [f1best,f2best] 

    "Representación gráfica"
    if unique==True:
        f1pa,f2pa = pareto_front()
        show_initial_graph(f1best,f2best,f1pa,f2pa,f1,f2)
    return poblacion,z, selector_cercanos,pesos,pens

def init16d(pob,xli,xui,peso,unique,seed):

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
    f1,f2,fitness,pens = evaluate_population16d(poblacion,peso)
    "Seleccionamos los mejores valores objetivos fi encontrados"
    f1best = find_best(f1)
    f2best = find_best(f2)
    z = [f1best,f2best] 

    "Representación gráfica"
    if unique==True:
        f1pa,f2pa = pareto_front()
        show_initial_graph(f1best,f2best,f1pa,f2pa,f1,f2)
    return poblacion,z, selector_cercanos,pesos,pens



def iterative4d(poblacion,xli,xui,selector_cercanos,pesos,f,z,peso):
    ''' MÉTODO ITERATIVO DEL ALGORITMO BASADO EN AGREGACIÓN'''

    '''PASO 1: SELECCIÓN ALEATORIA DE LOS ÍNDICES VECINOS DE CADA SUBPROBLEMA GENERAR UNA SOLUCIÓN CON OPERADORES
    EVOLUTIVOS'''
    "Nueva población de individuos"
    cr = 0.5
        
    poblacion,z,f1,f2,pens = operator4d(selector_cercanos,poblacion,f,xui,xli,cr,z,pesos,peso)
            
    return poblacion,z,f1,f2,pens

def iterative16d(poblacion,xli,xui,selector_cercanos,pesos,f,z,peso):
    ''' MÉTODO ITERATIVO DEL ALGORITMO BASADO EN AGREGACIÓN'''

    '''PASO 1: SELECCIÓN ALEATORIA DE LOS ÍNDICES VECINOS DE CADA SUBPROBLEMA GENERAR UNA SOLUCIÓN CON OPERADORES
    EVOLUTIVOS'''
    "Nueva población de individuos"
    cr = 0.5
        
    poblacion,z,f1,f2,pens = operator16d(selector_cercanos,poblacion,f,xui,xli,cr,z,pesos,peso)
    
    return poblacion,z,f1,f2,pens