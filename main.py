
from alg import init,normal_distribution, distance, generate_population, evaluate_population, find_best, select_random_index

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
poblacion, z, selector_cercanos = init(51,xli,xui)

print("Nuevos índices vecinos seleccionados: ")
res = select_random_index(selector_cercanos,poblacion,0.5,xui,xli)
print(res)