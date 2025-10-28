import torch
import torch.nn
from torch.nn import functional

device = "gpu" if torch.cuda.is_available() else "cpu"

#-----------------TOKENIZATION DEL DATASET-----------------------------

# Leer dataset y obtener lista de valores únicos dentro del dataset
with open(r"GptDesde0\DataSet.txt", "r", encoding="utf-8") as f:
    data = f.read()
unique_values = sorted(list(set(data)))

# Creamos nuestro diccionario de tokens y funciones de codificación/decodificación
value_to_number = {value: number for number, value in enumerate(unique_values)}
number_to_value = {number: value for number, value in enumerate(unique_values)}

encode = lambda text: [value_to_number[char] for char in text]
decode = lambda number_list: ''.join([number_to_value[number] for number in number_list])
