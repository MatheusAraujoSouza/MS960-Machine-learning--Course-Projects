# -*- coding: utf-8 -*-
"""
MS960 Unicamp
Raul Augusto Teixeira, RA 205177
Gabriel Borin Macedo, RA 197201
Matheus Araujo Souza, RA 184145
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
#leitura do arquivo
#Primeira parte fica com a abertura dos arquivos

guard = np.zeros((401,1), dtype=np.float64)
Mguard =np.zeros((401,10), dtype=np.float64)
Pguard =np.zeros((4999,10), dtype=np.float64)

dfy=pd.read_csv("labelMNIST.csv",header=None)
dfy.head()


#agora vamos definir os intervalos dos atributos
#vamos primeiro tentar diferenciar o 10 dos demais itens 
#e ver como minha funcao esta convergindo neles 
X=pd.read_csv("imageMNIST.csv",header=0,decimal=',').values
Xguard=pd.read_csv("imageMNIST.csv",header=0,decimal=',').values
y = dfy.iloc[:-1,0].values
Yguard=dfy.iloc[:-1,0].values
#Xguard foi criado para a utilizacao da funcao 
#plt.imshow(Xguard[2600,:].reshape(20,20), cmap='gray')
#mas so sera usada em outra janela 

#agora vamos separar cada tipo de exemplo, dado que vamos atribuir 
#positivo para 10 e negativo para qualquer outro diferente de 10
#pos,neg = (y==9).reshape(len(y),1),(y!=9).reshape(len(y),1)


#criando minha funcao sigmoid
#declarando minha função sigmoide 
def sigmoid(z):
    return 1/(1+np.exp(-z))
#tambem fazendo minha funcao de custo
#Minha J(teta) tambem conhecida como funcao de custo
#aqui estamos fazendo a parte vetorizada 
def computeCost(X,y,theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    J = (1/m)*np.sum((-y*np.log(h))-((1-y)*np.log(1-h)))
    return J

#agora vamos usar a gradiente descendente
#J_history vai armazenar meu histórico da funcao de custo 
#No caso len(y) vai calcular o numero de exemplos no conjunto de treinamento
#aqui tambem definimos o nbr_iter como o numero total de iteracoes
def gradientDescent (X,y,theta,alpha,nbr_iter):
    J_history = []
    m = len(y)
    
    for i in range(nbr_iter):
        h = sigmoid(X.dot(theta))
        theta = theta - (alpha/m)*(X.T.dot(h-y))
        J_history.append(computeCost(X,y,theta))
        
    return theta,J_history

#aqui vamos criar nossa funcao de prediction
def prediction(X,new_theta):
    pred = sigmoid(np.dot(X,new_theta))
    return pred
#preciso definir o numero de exemplos de treinamento
#m e o numero do meu exemplo de treinamento 
#n o numero de atributos

m = X.shape[0]
n = X.shape[1]

#aqui vamos colocar a primeira coluna como tudo 1
X = np.append(np.ones([m,1]),X,axis=1)

#para meu y vamos transformar em um vetor coluna
y = np.array(y).reshape(-1,1)
Yguard=np.array(y).reshape(-1,1)
#definindo um valor para o teta
theta = np.zeros([n+1,1])
cost = computeCost(X,y,theta)
#vai printar o custo inicial
print(cost)
nbr_iter = 10
alpha = 1.5
#aqui estamos fazendo o loop para a verificacao de cada probabilidade
#caminhando por todo o nosso vetor de teste do 0 ao 4998
#geramos entao nossos thetas e assim criando uma matriz de probabilidade para cada linha
#a maior probabilidade e' o valor que ele indica como sendo o possivel numero
#esse e o motivo que usamos o argmax em Pguard 

for g in range(0,4999):
    for i in range(1,11):  
   
        for j in range(0,4999):
            if y[j,0] == i:
                y[j,0] = 1
            else:
                y[j,0] = 0

        new_theta,J_history = gradientDescent(X,y,theta,alpha,nbr_iter)
        new_cost = computeCost(X,y,new_theta)
        prob = prediction(X[g,:],new_theta)
        #aqui vamos voltar os valores de y
        for k in range(0,4999):
            y[k,0]=Yguard[k,0]
        Pguard[g,i-1]=prob
      


confusion = confusion_matrix(Yguard, np.argmax(Pguard,axis=1))

















