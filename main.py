
from alg import init,iterative
import matplotlib.pyplot as plt

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
    poblacion,z,sol1,sol2 = main(individuos,generaciones,f)
    print("Población final: ",poblacion)

    plt.scatter(z[0],z[1], color='red')
    plt.scatter(sol1,sol2)
    plt.show()

out(100,100,0.5)