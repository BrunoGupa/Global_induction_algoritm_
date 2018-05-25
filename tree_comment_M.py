#ALGORTIMO EVOLUTIVO DE INDICCION GLOBAL EN ARBOLES DE DECISION
###En la primera parte del codigo se diseña un arbol de decision basado en el metodo CART
#Esta primera parte fue tomada del blog: MACHINE LEARNING MASTERY
#La publicacion es "How to implement the decision tree algorithm from scratch in python"
#y esta disponible en la siguiente liga:
#https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#
#En la segunda parte, se desarrolla un algoritmo genetico a partir de los arboles de decision.
#Esta parte se basa en el blog mencionado anteriormente y en el articulo 
#M. Kretowski and M. Czajkowski. An evolutionary algorithm for global induction
#of regression trees. In L. Rutkowski, R. Scherer, R. Tadeusiewicz, L. A.
#Zadeh, and J. M. Zurada, editors, Artificial Intelligence and Soft Computing,
#pages 157-164, Berlin, Heidelberg, 2010. Springer Berlin Heidelberg
#
#
###PARTE 1####


# CART sobre el cojunto de datos Banknote autentication
from random import seed
from random import randrange
import random
from csv import reader
import numpy as np

# Cargar el archivo CSV
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Conversion del string columna a flotante
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Dividir los datos entre el conjunto de entramiento y el conjunto de prueba
def split_dataset(dataset, train_size):
	dataset_split = list()
	dataset_copy = list(dataset)
	test_size = int(len(dataset)*(1-train_size))
	test_set = list()
	while len(test_set) < test_size:
		index = randrange(len(dataset_copy))
		test_set.append(dataset_copy.pop(index))
	train_set = dataset_copy
	dataset_split.append(train_set)
	dataset_split.append(test_set)
	return dataset_split

# Calculo del porcentaje del error de clasificacion erronea entre la muestra actual y la muestra a predecir (accuracy).
def accuracy_metric(actual, predicted):
	incorrect = 0
	for i in range(len(actual)):
		if actual[i] != predicted[i]:
			incorrect += 1
	return incorrect / float(len(actual)) * 100.0

# Evaluacion del algoritmo de induccion global usando validacion cruzada y accuracy.
def evaluate_algorithm(dataset, algorithm, training_size, *args):
	dataset_split = split_dataset(dataset, train_size)
	train_set = list(dataset_split[0])
	test_set = list(dataset_split[1])
	best_tree = algorithm(train_set, test_set, *args)

	target = [row[-1] for row in test_set]
	tree_prediction = list()
	for row in test_set:
		prediction = predict(best_tree, row)
		tree_prediction.append(prediction)
	error = accuracy_metric(target, tree_prediction)
	return error, train_set, test_set

# Misma Eevaliacon pero para el algortimo CART tradicional
def evaluate_algorithm_cart(train_set, test_set, cart_tree, train_size, max_depth, min_size):
	train_set = list(train_set)
	test_set = list(test_set)
	best_tree = cart_tree(train_set, max_depth, min_size)
	target = [row[-1] for row in test_set]
	tree_prediction = list()
	for row in test_set:
		prediction = predict(best_tree[0], row)
		tree_prediction.append(prediction)
	error = accuracy_metric(target, tree_prediction)
	return error

# Divide al conjunto de datos con base a un atributo y a determinado valor del atributo
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Se calcula el indice Gini para la base de datos divida en grupos
def gini_index(groups, classes):
	# se cuenta el numero de muestras en todos los grupos. 
	n_instances = float(sum([len(group) for group in groups]))
	# suma ponderada del indice Gini en cada grupo
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# se evita la division entre cero
		if size == 0:
			continue
		score = 0.0
		# se cuenta el numero de elementos de cada clase,
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# despues de multiplicarse entre si, se pondera y se resta con 1 para calcular el Gini
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Se selecciona la mejor división para el conjunto de datos
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Creacion del nodo termial (hoja)
# Esta se crea a partir de la etiqueta que mas se repita en el subconjunto de datos.
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Creacion de los hijos (dos, uno o cero) por cada nodo. 
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# se revisa si hay nodos sin division 
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# Se comprueba si se ha alcanzado la profundidad maxima
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# Proceso el hijo izquierdo 
			#si el nodo ha alcanzado el tamano minimo de datos, se convierte en hoja.
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		#de otro modo, se divide. 
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# Proceso del hijo derecho
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Para construir el arbol de decision se toman todos los datos para apartir de ahi or dividiendolos.
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Hacer predicciones con el arbol de decision.
# A partir de un nodo y una fila (vector de datos) 
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


#### PARTE 2 ######


#### POBLACION INICIAL ####
# Se toman n muestras aleatorias del conjunto de entrenamiento
# *n_trees = numero de partes en que se dividiran
# *train = conjunto de datos de entrenamiento
def split_train_set(train, n_trees):
	train_split = list()
	train_copy = list(train)
	split_size = min(int(len(train)*0.1),500)
	for tree in range(n_trees):
		split = list()
		index = np.random.choice(len(train_copy), size = split_size, replace=False, p=None)
		for i in index:
			split.append(train_copy[i])
		train_split.append(split)
	return train_split

#A partir de cada submuestra de entrenamiento, se crea un arbol.
# *train_split = conjunto de submuestras
# *max_depth = profundidad máxima del arbol (del nodo raiz a las hojas)
# *min_size = es el numero minimo de elementos que puede haber en una hoja. 
#             Si se alcanza este numero se deja de dividir, aun cuando la profundidad del arbol aumente (nodos con un hijo).
def get_trees(train_split, max_depth, min_size):
	trees = list()
	for i in range(len(train_split)):
		trees.append(build_tree(train_split[i], max_depth, min_size))
	return trees


#### FUNCION DE APTITUD ####
#Se pasa toda la muestra de entrenamiento a traves de cada elemento de la poblacion.
def eval_trees(train, trees):
	trees_evals = list()
	# Se toma la etiqueta de cada vector de datos del conjunto de entremaniento.
	real = [row[-1] for row in train] 
	# Se mide la exatitud (accuracy) entre las clasificaciones del arbol y las etiquetas reales.
	for i in range(len(trees)):
		tree_evals = list()
		for row in train:
			eval = predict(trees[i], row)
			tree_evals.append(eval)
		trees_evals.append(accuracy_metric(real, tree_evals))
	return trees_evals

#Se crea una lista ordenada de cada arbol con su evaluacion
def sorted_duple(trees, evals):
	duples = list()
	for i in range(len(trees)):
		duples.append([trees[i],[evals[i]]])
	sorted_duples = sorted(duples, key = lambda duple: duple[1])
	return sorted_duples


#### SELECION ####
# Se selecciona a los dos padres por jerarquia lineal
# * poblacion_size = tamano de la poblacion inicial
# * number_max = define el peso de las probabilidades del vector. 1 < number_max < 2. 
def choice(duples1, number_max = 1.7):
	#se renombra la lista para no hacer modificaciones en la lista orininal.
	duples = list(duples1)
	#index = randrange(len(train_copy))
	poblacion_size = len(duples)
	# lista donde se guardaran los dos padres
	samples = list()
	# number_max es un parametro libre entre 1 y 2 con valor predeterminado 1.7	
	number_min = 2 - number_max
	# aqui cada arbol se etiqueta con el elemento de la lista ordenada duples1
	# la jerarquia se toma en orden decreciente de aptitud (primero el peor elemento)
	def jerarquia(i):
		return poblacion_size + 1 - i
	def Valesp(i):
		value = number_min + (number_max - number_min) * (float( jerarquia(i) -1) / (poblacion_size - 1) )
		return value
	probabilidades = list()
	#Se calcula el peso con el que debe ponderarse la probabilidad 
	peso = 0
	for i in range(1,poblacion_size + 1):
		peso = peso + Valesp(i)
	# Se actualiza el vector de probabilidades ponderando con el peso
	for i in range(1,poblacion_size + 1):
		probabilidades.append(Valesp(i) / peso) 
	# Se seleccionan los dos padres por medio de rueda de ruleta
	index = np.random.choice(range(0,poblacion_size), 2, p = probabilidades,replace=False)
	#samples = np.random.choice(duples, 2, p=[0.4, 0.3, 0.2, 0.075, 0.025], replace=False)
	samples.append(duples[index[0]])
	samples.append(duples[index[1]])
	return samples


##### CRUZA ####
# Se elige un sub-arbol aleatoriamente de cada padre. 
def random_node(sample,max_depth):
	tree = dict(sample)
	# se obtiene un numero aleatorio entre 1 y max_depth
	random_depth = random.randint(1, max_depth)
	# se disena un camino aleatorio (sucesion de izquerda o derecha) a traves de las ramas con profundidad max_depth.
	for level in range(random_depth):
		right = random.randint(0, 1)
		if (right):
			if isinstance(tree['right'], dict):
				tree = tree['right']
			else:
				tree = tree['right']
				return tree
		else:
			if isinstance(tree['left'], dict):
				tree = tree['left']
			else:
				tree = tree['left']
				return tree
	# el camino elegido caracteriza al nodo; de donde se obtiene el sub-arbol.
	return tree

# Creacion de los desendientes a partir de intercambiar sus sub-arboles.
def new_tree(sample,max_depth,random_node):
	import copy
	tree = copy.deepcopy(sample)
	new_tree = copy.deepcopy(sample)
	path = list()
	random_depth = random.randint(1, max_depth)
	# se crea alatoriamente la profundidad del camino.
	for level in range(random_depth):
		right = random.randint(0, 1)
		if (right):
			path.append('right')
			if isinstance(tree['right'], dict):
				tree = tree['right']
			else:
				tree = tree['right']
				#return tree
				break
		else:
			path.append('left')
			if isinstance(tree['left'], dict):
				tree = tree['left']
			else:
				tree = tree['left']
				#return tree
				break

	lista = new_tree
	for i in range(len(path)):
		if i == len(path) -1:
			lista[path[i]] = random_node
		else:
			lista = lista[path[i]]

	return new_tree




#Creacion de los dos hijos
def crossover(samples1,max_depth):
	samples = list(samples1)
	random_nodes = list()
	new_trees = list()
	
	#copiar un nodo aleatrorio de la muestra
	for i in range(len(samples)):
		random_nodes.append(random_node(samples[i][0],max_depth))
	
	#sustituir los nodos aleatorios
	new_trees.append(new_tree(samples[0][0],max_depth,random_nodes[1]))
	new_trees.append(new_tree(samples[1][0],max_depth,random_nodes[0]))
	
	return new_trees


#### MUTACION ####
# A partir de una muestra de tamano 100 del conjunto de entrenamiento se crea un conjunto de splits
def set_midvalues(train):
	train_copy = list(train)
	midvalues = list()
	for feature in range(len(train_copy[0])-1):
		column = list()
		for instance in range(len(train_copy)):
			column.append(train_copy[instance][feature])
		column_sorted = sorted(column)
		#de la lista ordenada se guardan los puntos medios. 
		midvalue = list()
		for instance in range(len(column_sorted)-1):
			midvalue.append((column_sorted[instance] + column_sorted[instance+1]) / 2)
		midvalues.append(midvalue)
	return midvalues


#Se selecciona un nuevo punto de division para el nodo a mutar
def get_newsplit(midvalues, index, value):
	list_mid = list(midvalues)
	list_mid[index].append(value)
	list_mid[index] = sorted(list_mid[index])
	value_index = list_mid[index].index(value)
	up = random.randint(0, 1)
	if up:
		if value_index == len(list_mid[index])-1:
			new_value = list_mid[index][0]
		else:
			new_value = list_mid[index][value_index+1]
	else:
		if value_index == 0:
			new_value = list_mid[index][len(list_mid[index])-1]
		else:
			new_value = list_mid[index][value_index-1]

	return new_value
	
# Se selecciona un nodo al azar y se elige un nuevo punto de division respetando el atributo que tenia
def new_tree_mutation(tree,max_depth,midvalues):
	import copy
	tree = copy.deepcopy(tree)
	new_tree = copy.deepcopy(tree)
	path = list()
	# Se escoge una profundidad al azar
	random_depth = random.randint(1, max_depth)

	for level in range(random_depth):
		right = random.randint(0, 1)
		if (right):
			path.append('right')
			if isinstance(tree['right'], dict):
				tree = tree['right']
			else:
				tree = tree['right']
				break
		else:
			path.append('left')
			if isinstance(tree['left'], dict):
				tree = tree['left']
			else:
				tree = tree['left']
				break

	#Se recorre el camino guardado para mutar al nuevo individuo
	lista = new_tree
	for i in range(len(path)):
		if i == len(path) -1:
			#si el nodo elegido no es una hoja
			if isinstance(lista[path[i]], dict):
				index = lista[path[i]]['index']
				value = lista[path[i]]['value']
				lista[path[i]]['value'] = get_newsplit(midvalues,index, value)	
			else:
				index = lista['index']
				value = lista['value']
				lista['value'] = get_newsplit(midvalues,index, value)   
		else:
			lista = lista[path[i]]

	return new_tree



# Se realiza la mutacion
def mutation(trees,max_depth,midvalues):
	new_trees = list()
	for tree in trees:
		new_trees.append(new_tree_mutation(tree,max_depth,midvalues))
	return new_trees


		
	
# Ejecucion del ALGORITMO DE INDUCCION GLOBAL
def decision_tree(train, test, max_depth, min_size, n_trees):
	midvalues = set_midvalues(train)
	train_split = split_train_set(train, n_trees)
	trees = get_trees(train_split, max_depth, min_size)
	evals = eval_trees(train, trees)
	duples = sorted_duple(trees, evals)
	bestfitness = duples[0][1]
	cont = 0
	cont2 = 0
	while(cont < 5000 and duples[0][1] > [0] and cont2 <1000):
		bestfitness = duples[0][1]
		selected_samples = choice(duples)
		crossover_samples = crossover(selected_samples,max_depth)
		mutated_samples = mutation(crossover_samples,max_depth,midvalues)
		eval_samples = eval_trees(train, mutated_samples)

		for mutated_tree in mutated_samples:
			trees.append(mutated_tree)
		for evalu in eval_samples:
			evals.append(evalu)

		duples = sorted_duple(trees, evals)
		duples = duples[0:n_trees]

		cont += 1
		if duples[0][1] == bestfitness:
			cont2 += 1
		else:
			cont2 = 0

		if (cont%200) == 0:
			print(cont, duples[0][1][0])
	print(cont, duples[0][1][0])
	print("*********************************")
	return duples[0][0]

# Ejecición del ALGORITMO CART (con fines comparativos)
def cart_tree(train1, max_depth, min_size):
	#= split_train_set(train, n_trees)
	train = [train1]
	trees = get_trees(train, max_depth, min_size)
	return trees

# Se califica la aptitud del CART en la base de datos Bank Note
# Carga y preparacion de los datos 
filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)
# convertir los atributos strings a enteros
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)

#### Especificaciones del algoritmo #####
train_size = .70
max_depth =5
min_size = 10
n_trees = 50
########################################

error, train_set, test_set = evaluate_algorithm(dataset, decision_tree, train_size, max_depth, min_size, n_trees)
print('Error Genetic CART: %s' % error)

error_cart = evaluate_algorithm_cart(train_set, test_set, cart_tree, train_size, max_depth, min_size)
print('Error CART: %s' % error_cart)

