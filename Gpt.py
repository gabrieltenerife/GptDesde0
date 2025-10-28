import torch
import torch.nn
from torch.nn import functional

dispositivo = "gpu" if torch.cuda.is_available() else "cpu"

#-----------------TOKENIZACION DEL DATASET-----------------------------

#Leer datase y btener lista de valores unicos dentro del dataset
with open("GptDesde0\DataSet.txt", "r", encoding="utf-8") as d:
    data = d.read()
valoresUnicos = sorted(list(set(data)))

#Creamos nuestro diccionario de tokens y funciones codificaccion/descodificaccion

ValorNumero = {valor: numero for numero, valor in enumerate(valoresUnicos)}
NumeroValor = {numero: valor for numero, valor in enumerate(valoresUnicos)}

codificar = lambda texto : [ValorNumero[caracter] for caracter in texto]
descodificar = lambda lista_numeros : ''.join([NumeroValor[numero] for numero in lista_numeros])

