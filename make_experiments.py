import json
from random import shuffle
from itertools import product

# Parâmetros da bateria de testes
tamanho = [10,50,100,150]
N = [10**4,10**5,10**6,10**7]
T = [50,20,10]
epsilon = [1,2]
K = [6,8,10,12]

tests = [valor for valor in product(tamanho,N,T,K,epsilon)]
shuffle(tests)

with open('experimentos.json','w') as fil:
    json.dump(tests,fil)


