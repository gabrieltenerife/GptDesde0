import torch
import torch.nn as nn
from torch.nn import functional

device = "cuda" if torch.cuda.is_available() else "cpu"

#-----------------TOKENIZATION DEL DATASET-----------------------------
#En esta parte del codigo, se sacan todos los caracteres de nuestro dataset y se les asigna
#un valor numeral "token". Ademas se crean los metodos "Caracter" -> "token" y viceversa 


# Leer dataset y obtener lista de valores únicos dentro del dataset
with open(r"GptDesde0\DataSet.txt", "r", encoding="utf-8") as f:
    data = f.read()

vocab = sorted(list(set(data)))

# Creamos nuestro diccionario de tokens y funciones de codificación/decodificación
value_to_number = {value: number for number, value in enumerate(vocab)}
number_to_value = {number: value for number, value in enumerate(vocab)}

encode = lambda text: [value_to_number[char] for char in text]
decode = lambda number_list: ''.join([number_to_value[number] for number in number_list])

#-----------------EMBEDDING "vectorizar tokens"-----------------------------
#En este codigo se transforman los tokens en vectores, que definen no solo cada uno de los
#caracteres de nuestro diccionario, tambien su posicion en un imput. Entre otras cosas esto define la capacidad del modelo.

# vocab_size: número total de tokens posibles (tamaño del vocabulario).
# vector_dimension: dimensión del vector que representa cada token (profundidad del embedding).
# block_size: longitud máxima de secuencia que el modelo puede ver a la vez (ventana de contexto).

# -> vector_dimension cuántas columnas (tamaño de cada vector),
# -> block_size cuántos tokens puede procesar simultáneamente.
#Total parámetros de embeddings = vocab_size * vector_dimension + block_size * vector_dimension

block_size = 256
vector_dimension = 256
vocab_size = len(vocab)

#Tablas embedding
token_embedding_table = nn.Embedding(vocab_size, vector_dimension)
position_embedding_table = nn.Embedding(block_size, vector_dimension)