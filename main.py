
from alg import init,normal_distribution, distance, generate_population, evaluate_population, find_best

'''
l = normal_distribution(0.1,11)
distances = distance(l)
print("Distancias euclídeas: " )
print(distances)
print(len(distances))




poblacion = generate_population(11, xli, xui)

print("Fitness: ")
f1,f2,evaluated_population = evaluate_population(poblacion)
print(f1)
minimos = find_best(f1)
print(minimos)
'''
"Creación del espacio de búsqueda"
xli = [ 0 for i in range(0,30)]
xui = [1 for i in range(0,30)]
poblacion, z = init(11,xli,xui)
