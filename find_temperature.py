import numpy as np

from sa_tsp import Custo, Pertubacao

def EncontrarTemperatura(posicoes,T0=100,limiar=.9,decaimento=0.9,rodadas=1000,Custo=Custo,Pertubacao=Pertubacao):

    desempenho = 1.0
    passo = 0
    x = np.arange(posicoes.shape[1])
    jx = Custo(x,posicoes)
    while desempenho > limiar:
        T = (decaimento**passo)*T0
        passo += 1
        desempenho = MedirAceitacao(x,jx,posicoes,T,rodadas,Custo,Pertubacao)
    return T

def MedirAceitacao(x,jx,posicoes,T,rodadas,Custo,Pertubacao):
    aceitos = 0
    for i in range(rodadas):
        xhat = Pertubacao(x)
        jhat = Custo(xhat,posicoes)
        if np.random.uniform(0,1) < np.exp((jx-jhat)/T):
            aceitos += 1
    return aceitos/rodadas

if __name__ == "__main__":
    import pandas as pd
    from generate_tsp import GerarProblemaRadialTSP
    import seaborn as sns
    import matplotlib.pyplot as plt 
    tamanho = 10
    
    aceitacoes = pd.DataFrame([],columns=['Tempeture','Acceptance rate'])
    for i in range(3):
        posicoes = GerarProblemaRadialTSP(tamanho)
        x = np.arange(posicoes.shape[1])
        jx = Custo(x,posicoes)
        aceitacao = {'Temperature':[],'Acceptance rate':[]}
        for T in np.linspace(.01,20,10):
            aceitacao['Temperature'].append(T)
            aceitacao['Acceptance rate'].append(MedirAceitacao(x,jx,posicoes,T,1000,Custo,Pertubacao))
        aceitacoes = pd.concat([aceitacoes,pd.DataFrame(aceitacao)],ignore_index=True)
    aceitacoes.to_pickle('aceitacoes.pickle')
    print(aceitacoes['Temperature'])
    sns.lineplot(aceitacoes,x='Temperature',y='Acceptance rate') 
#    plt.xscale('log')
    plt.show()
