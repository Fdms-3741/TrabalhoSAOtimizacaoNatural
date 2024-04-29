import json
from multiprocessing import Pool
import time as tm 
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Funções para o Simulated Annealing
from sa_tsp import SimulatedAnnealingTSP, Custo, PertubacaoSwitch, PertubacaoLin
# Geradores de problemas com soluções conhecidas
from generate_tsp import GerarProblemaRadialTSP, GerarProblemaRetangularTSP
# (Falho) Tentativa de encontrar temperatura
from find_temperature import EncontrarTemperatura
# Gerador de problemas com base na lista do TSPLIB
from tsplib_problems import GerarProblemaTSPLIB

def Done(res):
    print(f"DONE: pos={res['Posições'].shape}, K={res['K']}, N={res['N']}")
    return res

def IterarExperimentos(problemas,Ks,Ns,T0s,epsilons,gaps=[0.99999999],Pertubacoes=[PertubacaoLin]):
    """
    Itera entre inúmeros parâmetros para tentar solucionar o problema
    """

    problemas = [problemas] if type(problemas)!=list else problemas
    # Conversão pra listas
    Ks = [Ks] if type(Ks)!=list else Ks
    Ns = [Ns] if type(Ns)!=list else Ns
    gaps = [gaps] if type(gaps)!=list else gaps
    T0s = [T0s] if type(T0s)!=list else T0s
    epsilons = [epsilons] if type(epsilons)!=list else epsilons
    Pertubacoes = [Pertubacoes] if type(Pertubacoes)!=list else Pertubacoes
    # Iteração entre diferentes experimentos
    
    resultados = {}

    with Pool(processes=6) as pool:
        tarefas = []
        # Executa as tarefas
        for problema, K, N, T0, epsilon, gap, Pertubacao in product(problemas,Ks,Ns,T0s,epsilons,gaps,Pertubacoes):
            args = (problema[0],K,N,T0,epsilon,problema[1])
            kwargs = {"gap":gap,"Pertubacao":Pertubacao}
            tarefas.append(pool.apply_async(ExperimentoTSP,args,kwargs,callback=Done))
            
        saidasFuncoes = [i.get() for i in tarefas]
        for resultado in saidasFuncoes:
        # Concatenação do resultado
            for key in resultado.keys():
                if key in resultados.keys():
                    resultados[key].append(resultado[key])
                else:
                    resultados[key] = [resultado[key]]

    return resultados

def ExperimentoTSP(posicoes,K,N,T0,epsilon,valorOtimo=None,gap=0.999999,Pertubacao=PertubacaoLin):
    # Calcula solução e custo da solução
    x0 = np.arange(posicoes.shape[1])
    
    # Escolhe vetor aleatório
    np.random.shuffle(x0)
    j0 = Custo(x0,posicoes)
    
    # Salva resultados
    xmin, jmin, jhist, thist, jminhist, exphist, tempoTotal, histTransicao, parada, xTransicao = SimulatedAnnealingTSP(x0,posicoes,K,N,T0,epsilon,valorOtimo,gap=gap,Pertubacao=Pertubacao)
    
    resultados = {}
    resultados['K'] = K
    resultados['N'] = N 
    resultados['$T_0$'] = T0 
    resultados['$\epsilon$'] = epsilon
    resultados['Posições'] = posicoes
    resultados['Valor ótimo'] = valorOtimo
    resultados['X'] = xmin
    resultados['J'] = jmin
    resultados['Condição de parada'] = parada
    resultados['Tempo total'] = tempoTotal 

    resultados['$X_0$'] = x0
    resultados['$J_0$'] = j0
    resultados['Evolução J'] = jhist
    resultados['Evolução T'] = thist
    resultados['Evolução $J_{min}$'] = jminhist
    resultados['Evolução aceitação'] = exphist
    resultados['Transições'] = histTransicao
    resultados['Transições X'] = xTransicao
    resultados['Gap'] = jmin/valorOtimo if valorOtimo else None
    
    return resultados

def ExperimentoRadialTSP(tamanho,K,N,T0,epsilon,gap=0.99999,Pertubacao=PertubacaoLin):
    posicoes = GerarProblemaRadialTSP(tamanho)
    valorOtimo = Custo(np.arange(posicoes.shape[1]),posicoes)
    print(posicoes)
    resultados = IterarExperimentos([(posicoes,valorOtimo)],K,N,T0,epsilon,gaps=gap,Pertubacoes=Pertubacao)
    return resultados

def ExperimentoRetangularTSP(ladoCentros, cidades, K,N,T0,epsilon,gap=0.99999):
    posicoes = GerarProblemaRetangularTSP(ladoCentros, cidades)
    valorOtimo = None 
    resultados = IterarExperimentos([(posicoes,valorOtimo)],K,N,T0,epsilon,gaps=gap,Pertubacoes=PertubacaoLin)
    return resultados

def ExperimentoTSPLIB(indices, K,N,T0,epsilon,gap=0.99999,pertubacoes=[PertubacaoLin]):
    problemas = []
    for i in indices:
        caracteristicas, posicao = GerarProblemaTSPLIB(i)
        problemas.append((posicao.T,caracteristicas['OPTIMAL_VALUE']))
    resultados = IterarExperimentos(problemas,K,N,T0,epsilon,gaps=gap,Pertubacoes=pertubacoes)
    return resultados

if __name__ == "__main__":
    import sys 
    # 
    ladoCentros = 4
    cidadesCentros = 3
    tamanho = (ladoCentros**2)*cidadesCentros
    # 
    K = [10**1,10**2,10**3]
    N = [10**3,10**4]
    T0 = [10]
    epsilon = 1

    
#    a = ExperimentoRetangularTSP(ladoCentros,cidadesCentros, K, N, T0, epsilon)
#    b = ExperimentoRadialTSP(tamanho, K, N, T0, epsilon,0.9999,PertubacaoLin)
    resultados = ExperimentoTSPLIB([0,1],K,N,T0,epsilon,0.99999,PertubacaoLin)    

    resultados = pd.DataFrame(resultados)

    # Salva os resultados 
    if len(sys.argv) == 2:
        nomeArquivo = f'./resultados/{sys.argv[1]}.pickle'
    else:
        nomeArquivo = './resultados/resultados.pickle'

    resultados.to_pickle(nomeArquivo)
