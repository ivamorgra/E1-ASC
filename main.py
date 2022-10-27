from cProfile import label
from turtle import color
from alg import init,iterative, pareto_front
import matplotlib.pyplot as plt
import numpy as np
from alg import distance, normal_distribution, t_nearest, get_index
"Para la creación de ficheros"
import os 

def create_files(individuos,generaciones,f,seed):
    output_all_file = open("./outputfiles/zdt3_all_"+"pob"+str(individuos)+"g"+str(generaciones)+"seed"+str(seed)+".out", "w")
    output_file = open("./outputfiles/zdt3"+"pob"+str(individuos)+"g"+str(generaciones)+"seed"+str(seed)+".out", "w")
    return output_all_file,output_file


def show_graph(individuos,generaciones,sol1,sol2,z,unique,seed):
    f1,f2 = pareto_front()
    plt.scatter(z[0],z[1], color='yellow',label='Best')
    plt.scatter(f1,f2,color='red',label='Pareto Front')
    if unique == True:
        f = open("./inputfiles/zdt3_final_popp200g50_seed01.out", "r")
        lines = f.readlines()
        print(lines[0])
        solnsga = []
        sol2nsga = []
        for l in lines[2:]:
            
            el = l.split("\t")
            solnsga.append(float(el[0]))
            sol2nsga.append(float(el[1]))

        f.close()
        plt.scatter(solnsga,sol2nsga, color='purple',label = "Solution from NSGA-II")
    
    plt.scatter(sol1,sol2, color='green',label='Solution from agregation')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.legend()
    if unique == True:
        plt.title('ZDT3 con '+str(individuos)+' individuos y '+str(generaciones)+' generaciones con seed aleatoria')
        plt.savefig('./ejecutions/zdt3/zdt3'+str(individuos)+'individuos'+str(generaciones)+'.png')
    else:
        plt.title('ZDT3 con '+str(individuos)+' individuos y '+str(generaciones)+' generaciones con seed '+str(seed))
    plt.savefig('./ejecutions/zdt3/zdt3'+str(individuos)+'individuos'+str(generaciones)+'generacionesseed'+str(seed)+'.png')
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
        output_all_file,output_file = create_files(individuos,generaciones,f,seed)
        "Iteración"
        for j in range(0,generaciones):
            
            poblacion,z,sol1,sol2 = iterative(poblacion,xli,xui,selector_cercanos,pesos,f,z)
            
            for s1,s2 in zip(sol1,sol2):
                s1cientific = np.format_float_scientific(s1, precision = 6, exp_digits=2)
                s2cientific = np.format_float_scientific(s2, precision = 6, exp_digits=2)
                zero = np.format_float_scientific(0, precision = 6, exp_digits=2)
                output_all_file.write(str(s1cientific)+"\t"+str(s2cientific)+"\t"+str(zero)+"\n")
                if j == generaciones-1:
                    output_file.write(str(s1cientific)+"\t"+str(s2cientific)+"\t"+str(zero)+'\n')
            
        
        show_graph(individuos,generaciones,sol1,sol2,z,False,seed)
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
        
    
        show_graph(individuos,generaciones,sol1,sol2,z,unique,0)
        
    
out(200,50,0.5,True)
