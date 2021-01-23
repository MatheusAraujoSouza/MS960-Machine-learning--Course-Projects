# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:06:01 2020

@author: 55199
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

x = np.array([40,50,60,70,80,90])

y = np.array([[10.607521697203472, 12.559999999999999, 26.934673366834172, 23.793490460157127, 10.195496164315763, 17.80791462872388 ],
              [9.733333333333333, 11.363636363636363, 24.95049504950495, 20.41958041958042, 9.591836734693878, 14.960629921259844],
              [9.396914446002805, 11.198738170347003, 25.07462686567164, 23.1622746185853, 8.742004264392325, 16.129032258064516]])

plt.figure(0)   #Plotagem dos dados de treinamento e aproximação
plt.plot(x,y[0,:],'-ok',c='red',marker='.',label='Dados de Treinamento')
plt.plot(x,y[1,:],'-ok',c='blue',marker='.',label='Dados de Validacao')
plt.plot(x,y[2,:],'-ok',c='green',marker='.',label='Dados de Teste')

plt.ylabel('Porcentagem de Acuracia')
plt.xlabel('Fração do Conjunto de Treinamento')
plt.legend()
plt.title('Precisao da Rede Neural em Funcao da Porcentagem do Grupo de Treinamento')