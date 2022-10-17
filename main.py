
from cProfile import label
from turtle import color
from alg import init,iterative, pareto_front
import matplotlib.pyplot as plt
import numpy as np
from alg import distance, normal_distribution, t_nearest, get_index
"Para la creación de ficheros"
import os 

def create_file(individuos,generaciones,f,seed):
    output_file = open("./outputfiles/zdt3"+"pob"+str(individuos)+"g"+str(generaciones)+"seed"+str(seed)+".out", "w")
    return output_file


def show_graph(individuos,generaciones,sol1,sol2,z,unique):
    f1,f2 = pareto_front()
    plt.scatter(z[0],z[1], color='yellow',label='Best')
    plt.scatter(f1,f2,color='red',label='Pareto Front')
    plt.scatter(sol1,sol2, color='green',label='Solution')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.legend()
    if unique == True:
        plt.title('ZDT3 con '+str(individuos)+' individuos y '+str(generaciones)+' generaciones con seed aleatoria')
    else:
        plt.title('ZDT3 con '+str(individuos)+' individuos y '+str(generaciones)+' generaciones con seed 9')
    plt.show()

''' CONSTRUCCIÓN DEL ALGORITMO PRINCIPAL'''
def main_zdt3_ficheros(individuos, generaciones,f):
    "Creación del espacio de búsqueda"
    xli = [ 0.0 for i in range(0,30)]
    xui = [1.0 for i in range(0,30)]

    "Inicialización del algoritmo"
    for seed in range(0,10):
        print("Ejecución con semilla: ",seed)
        poblacion, z, selector_cercanos,pesos = init(individuos,xli,xui,False,seed)
        output_file = create_file(individuos,generaciones,f,seed)
        "Iteración"
        for j in range(0,generaciones):
            
            poblacion,z,sol1,sol2 = iterative(poblacion,xli,xui,selector_cercanos,pesos,f,z)

            for s1,s2 in zip(sol1,sol2):
                output_file.write(str(s1)+"   "+str(s2)+"   "+"0.0"+"\n")
        
        output_file.close()
    
    return poblacion,z,sol1,sol2

def main(individuos, generaciones,f):
    "Creación del espacio de búsqueda"
    xli = [ 0.0 for i in range(0,30)]
    xui = [1.0 for i in range(0,30)]

    "Inicialización del algoritmo"
    poblacion, z, selector_cercanos,pesos = init(individuos,xli,xui,True,0)
    
    "Iteración"
    for j in range(0,generaciones):
            
        poblacion,z,sol1,sol2 = iterative(poblacion,xli,xui,selector_cercanos,pesos,f,z)

    return poblacion,z,sol1,sol2

def out(individuos,generaciones,f,unique):
    ''' Parámetros de entrada:
    - individuos: número de individuos de la población
    - generaciones: número de generaciones(iteraciones)
    -f: probabilidad de mutación
    -unique: booleano: True si queremos que el algoritmo se ejecute una sola vez sin tener en cuenta la semilla que queremos;
    False si queremos tener varias ejecuciones con diferentes semillas'''

    if unique==False:
        ''' VARIAS EJECUCIONES CON DIFERENTES SEMILLAS'''
        poblacion,z,sol1,sol2 = main_zdt3_ficheros(individuos,generaciones,f)
    
    else:
        ''' UNA ÚNICA EJECUCIÓN'''
        poblacion,z,sol1,sol2 = main(individuos,generaciones,f)
        
    
    show_graph(individuos,generaciones,sol1,sol2,z,unique)
        
    
out(50,200,0.5,True)
