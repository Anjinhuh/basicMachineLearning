import numpy as np

import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# Maior que 1.65
# Pesa mais que 60kg
# Tem cabelo grande?


mulher1=[1, 0, 1]
mulher2=[0, 1, 1]
mulher3=[1, 1, 0]

homem1=[1, 1, 0]
homem2=[0, 1, 0]
homem3=[1, 1, 1]

treino_x = [mulher1, mulher2, mulher3, homem1, homem2, homem3]

treino_y = [1, 1, 1, 0, 0, 0]

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

pessoaMisterio1 = [1, 0, 1]
pessoaMisterio2 = [1, 1, 0]
pessoaMisterio3 = [1, 0, 1]


teste_x = [pessoaMisterio1, pessoaMisterio2, pessoaMisterio3]

teste_y = [0, 0, 0]

previsoes = modelo.predict(teste_x)


taxaDeAcerto = accuracy_score(teste_y, previsoes)
homemOuMulher = modelo.predict(teste_x)
for i in homemOuMulher:
    if(i == 1):
        print('Mulher')
    else:
        print('homem')
print('Taxa de acerto!',int(taxaDeAcerto*100),'%')