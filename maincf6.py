
from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from algcf6 import *

def create_file4d(individuos,generaciones,f,seed):
    output_file = open("./outputfiles/cf6_4d_all_popmp"+str(individuos)+"g"+str(generaciones)+"seed"+str(seed)+".out", "w")
    return output_file

def create_file16d(individuos,generaciones,f,seed):
    output_file = open("./outputfiles/cf6_16d_all_popmp"+str(individuos)+"g"+str(generaciones)+"seed"+str(seed)+".out", "w")
    return output_file


def show_graph4d(individuos,generaciones,sol1,sol2,z,unique,seed):
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
        plt.title('CF6 con '+str(individuos)+' individuos y '+str(generaciones)+' generaciones con seed aleatoria')
    else:
        plt.title('CF6 con '+str(individuos)+' individuos y '+str(generaciones)+' generaciones con seed '+str(seed))
    plt.savefig('./ejecutions/cf6_4d/cf6_4d_all_popmp'+str(individuos)+'g'+str(generaciones)+'_seed'+str(seed)+'.png')
    plt.show()


def show_graph16d(individuos,generaciones,sol1,sol2,z,unique,seed):
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
        plt.title('CF6 con '+str(individuos)+' individuos y '+str(generaciones)+' generaciones con seed aleatoria')
    else:
        plt.title('CF6 con '+str(individuos)+' individuos y '+str(generaciones)+' generaciones con seed '+str(seed))
    plt.savefig('./ejecutions/cf6_16d/cf6_16d_all_popmp'+str(individuos)+'g'+str(generaciones)+'_seed'+str(seed)+'.png')
    plt.show()
    


def main_cf64d_ficheros(individuos, generaciones,f,peso):

    "Creación del espacio de búsqueda 4 dimensiones"


    xli = [ -2 for i in range(1,4)]
    xui = [2 for i in range(1,4)]
    xui.append(1)
    xli.append(0)
    xui.sort()
    xli.sort(reverse=True)

    for seed in range(0,10):
        "Inicialización del algoritmo"
        poblacion, z, selector_cercanos,pesos = init4d(individuos,xli,xui,peso,False,seed)
        output_file = create_file4d(individuos,generaciones,f,seed)
        "Iteración"
        
        for i in range(0,generaciones):
            
            poblacion,z,sol1,sol2 = iterative4d(poblacion,xli,xui,selector_cercanos,pesos,f,z,peso)
            for s1,s2 in zip(sol1,sol2):
                output_file.write(str(s1)+"   "+str(s2)+"   "+"0.0"+"\n")
    
        show_graph4d(individuos,generaciones,sol1,sol2,z,False,seed)
        output_file.close()
    return poblacion,z,sol1,sol2


def main_cf64d(individuos, generaciones,f,peso):

    "Creación del espacio de búsqueda 4 dimensiones"


    xli = [ -2 for i in range(1,4)]
    xui = [2 for i in range(1,4)]
    xui.append(1)
    xli.append(0)
    xui.sort()
    xli.sort(reverse=True)

    
    "Inicialización del algoritmo"
    poblacion, z, selector_cercanos,pesos = init4d(individuos,xli,xui,peso,True,0)
    
    "Iteración"
        
    for i in range(0,generaciones):
            
        poblacion,z,sol1,sol2 = iterative4d(poblacion,xli,xui,selector_cercanos,pesos,f,z,peso)
        
    
    return poblacion,z,sol1,sol2

def out_cf64d(individuos,generaciones,f,peso,unique):

    if unique == False:
        poblacion,z,sol1,sol2 = main_cf64d_ficheros(individuos,generaciones,f,peso)
    else:
        poblacion,z,sol1,sol2 = main_cf64d(individuos,generaciones,f,peso)
        show_graph4d(individuos,generaciones,sol1,sol2,z,unique,0)

def main_cf616d_ficheros(individuos, generaciones,f,peso):

    "Creación del espacio de búsqueda 4 dimensiones"


    xli = [ -2 for i in range(0,15)]
    xui = [2 for i in range(0,15)]
    xui.append(1)
    xli.append(0)
    xui.sort()
    xli.sort(reverse=True)

    "Inicialización del algoritmo"
    for seed in range(0,10):
        print("Ejecución con semilla: ",seed)
        poblacion, z, selector_cercanos,pesos = init16d(individuos,xli,xui,peso,False,seed)
        output_file = create_file16d(individuos,generaciones,f,seed)
        "Iteración"
    
        for i in range(0,generaciones):
        
            poblacion,z,sol1,sol2 = iterative16d(poblacion,xli,xui,selector_cercanos,pesos,f,z,peso)
    
            for s1,s2 in zip(sol1,sol2):
                output_file.write(str(s1)+"\t"+str(s2)+"\t"+"0.0"+"\n")
        
        show_graph16d(individuos,generaciones,sol1,sol2,z,False,seed)
        output_file.close()
    
    return poblacion,z,sol1,sol2

def main_cf616d(individuos, generaciones,f,peso):

    "Creación del espacio de búsqueda 4 dimensiones"


    xli = [ -2.0 for i in range(0,15)]
    xui = [2.0 for i in range(0,15)]
    xui.append(1.0)
    xli.append(0.0)
    xui.sort()
    xli.sort(reverse=True)

    "Inicialización del algoritmo"
    poblacion, z, selector_cercanos,pesos = init16d(individuos,xli,xui,peso,True,0)

    "Iteración"
    
    for i in range(0,generaciones):
        
        poblacion,z,sol1,sol2 = iterative16d(poblacion,xli,xui,selector_cercanos,pesos,f,z,peso)
    
    
    return poblacion,z,sol1,sol2

def out_cf616d(individuos,generaciones,f,peso,unique):

    if unique == False:
        poblacion,z,sol1,sol2 = main_cf616d_ficheros(individuos,generaciones,f,peso)
    else:
        poblacion,z,sol1,sol2 = main_cf616d(individuos,generaciones,f,peso)
        show_graph16d(individuos,generaciones,sol1,sol2,z,unique,0)
    
   


#out_cf64d(50,200,0.5,2000,True)
out_cf616d(20,200,0.5,2000,False)



