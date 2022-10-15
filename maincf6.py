
from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from algcf6 import *

def main_cf64d(individuos, generaciones,f):

    "Creación del espacio de búsqueda 4 dimensiones"


    xli = [ -2 for i in range(1,4)]
    xui = [2 for i in range(1,4)]
    xui.append(1)
    xli.append(0)
    xui.sort()
    xli.sort(reverse=True)

    "Inicialización del algoritmo"
    poblacion, z, selector_cercanos,pesos = init4d(individuos,xli,xui)

    "Iteración"
    
    for i in range(0,generaciones):
        
        poblacion,z,sol1,sol2 = iterative4d(poblacion,xli,xui,selector_cercanos,pesos,f,z)
    
    
    return poblacion,z,sol1,sol2

def out_cf64d(individuos,generaciones,f):
    poblacion,z,sol1,sol2 = main_cf64d(individuos,generaciones,f)
    print("Población final: ",poblacion)
    f1,f2 = pareto_front()
    plt.scatter(z[0],z[1], color='yellow',label='Best')
    plt.scatter(f1,f2,color='black',label='Pareto Front')
    plt.scatter(sol1,sol2, color='green',label='Solution')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.show()
    



def main_cf616d(individuos, generaciones,f):

    "Creación del espacio de búsqueda 4 dimensiones"


    xli = [ -2 for i in range(0,15)]
    xui = [2 for i in range(0,15)]
    xui.append(1)
    xli.append(0)
    xui.sort()
    xli.sort(reverse=True)
    
    "Inicialización del algoritmo"
    poblacion, z, selector_cercanos,pesos = init4d(individuos,xli,xui)

    "Iteración"
    
    for i in range(0,generaciones):
        
        poblacion,z,sol1,sol2 = iterative4d(poblacion,xli,xui,selector_cercanos,pesos,f,z)
    
    
    return poblacion,z,sol1,sol2

def out_cf616d(individuos,generaciones,f):
    poblacion,z,sol1,sol2 = main_cf64d(individuos,generaciones,f)
    print("Población final: ",poblacion)
    f1,f2 = pareto_front()
    plt.scatter(z[0],z[1], color='yellow',label='Best')
    plt.scatter(f1,f2,color='black',label='Pareto Front')
    plt.scatter(sol1,sol2, color='green',label='Solution')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.show()


#out_cf64d(20,200,0.5)
out_cf616d(20,200,0.5)


