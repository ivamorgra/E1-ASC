
from alg import normal_distribution, distance, generate_population, evaluate_population


l = normal_distribution(0.1,11)
distances = distance(l)
print("Distancias euclídeas: " )
print(distances)
print(len(distances))

"Creación del espacio de búsqueda"
xli = [ 0 for i in range(0,30)]
xui = [1 for i in range(0,30)]


poblacion = generate_population(30, xli, xui)
print("Fitness: ")
fitness = evaluate_population(poblacion)
print(l)
print(fitness)