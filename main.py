
from turtle import color
from alg import init,iterative
import matplotlib.pyplot as plt
import numpy as np

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

def out(individuos,generaciones,f):
    poblacion,z,sol1,sol2 = main(individuos,generaciones,f)
    print("Población final: ",poblacion)
    f1,f2 = pareto_front()
    plt.scatter(z[0],z[1], color='yellow')
    plt.scatter(f1,f2,color='red')
    plt.scatter(sol1,sol2)
    plt.xscale('linear')
    plt.yscale('linear')
    plt.show()
    
out(50,200,0.5)