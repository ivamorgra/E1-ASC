
from cProfile import label
from turtle import color
from algcf6 import init,iterative, pareto_front
import matplotlib.pyplot as plt
import numpy as np
from algcf6 import distance, normal_distribution, t_nearest, get_index
''' CONSTRUCCIÓN DEL ALGORITMO PRINCIPAL'''
def main(individuos, generaciones,f):
    "Creación del espacio de búsqueda"
    xli = [ 0 for i in range(0,30)]
    xui = [1 for i in range(0,30)]

    "Inicialización del algoritmo"
    poblacion, z, selector_cercanos,pesos = init(individuos,xli,xui)

    "Iteración"
    for i in range(0,generaciones):
        
        poblacion,z,sol1,sol2 = iterative(poblacion,xli,xui,selector_cercanos,pesos,f,z)
    
    
    return poblacion,z,sol1,sol2



def out(individuos,generaciones,f):
    #poblacion,z,sol1,sol2 = main(individuos,generaciones,f)
    #print("Población final: ",poblacion)
    f1,f2 = pareto_front()
    #plt.scatter(z[0],z[1], color='yellow',label='Best')
    plt.scatter(f1,f2,color='black',label='Pareto Front')
    #plt.scatter(sol1,sol2, color='green',label='Solution')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.show()
    
out(50,200,0.5)
