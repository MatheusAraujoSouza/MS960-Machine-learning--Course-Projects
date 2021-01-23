# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:06:01 2020

@author: 55199
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
start_time=time.time()
# A funcao load mat estamos fazendo um intercambio entre os dados do matlab
# entao ele vai abrir o arquivo ja linearizado ponto mat e assim
# Ja vou pedir minha matriz de dados tanto para a variavel x quanto para a variavel y


mat = loadmat("ex3data1.mat")
X = mat["X"]
y = mat["y"]

train_size = 0.7
validation_size = 0.15
test_size = 0.15

index = np.random.rand(len(X)) < train_size  # boolean index
Xtrain, Xaux = X[index], X[~index]  # index and it's negation
ytrain, yaux = y[index], y[~index]

index = np.random.rand(len(Xaux)) < validation_size/(1-train_size)  # boolean index
Xvalidation, Xtest = Xaux[index], Xaux[~index]  # index and it's negation
yvalidation, ytest = yaux[index], yaux[~index]

###########################################plot###################################################################
# nesse caso agora vamos exibir as u imagens que pegamos usando a funcao do plt no caso 100 imagens
# para o tamanho 20 20
'''
fig, axis = plt.subplots(10, 10, figsize=(8, 8))
for i in range(10):
    for j in range(10):
        axis[i, j].imshow(X[np.random.randint(0, 5001), :].reshape(20, 20, order="F"), cmap="hot")
        axis[i, j].axis("off")
'''

# _____####################################funcoes##############################################______########
# Algoritmo :
#
#
# aqui no basico como na regressão Logistica temos
# funcao sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#################################sigmoid no gradiente##########################################
def sigmoidGradient(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid * (1 - sigmoid)


###################################computando o custo#################################################
# Funcao de custo da rede e' a soma do custo de todas as K saidas:
# no caso nossos thetas ja vão estar linearizados para nossos calculos ficarem mais diretos
# input_layer_size: numero de elemnetos na camada de entrada
# hidden_layer_size: numero de elementos na camada escondida
# labda: esse é o labda proeminente da minha regularizacao
# num_labels:numero de elementos na camada de saida o numero de classes
##################################################################################################


def computeCost(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda):
    #############################################comentarios#####################################################
    # Theta 1 e Theta 2
    # No caso no inicio vamos criar duas matrizes de peso  theta1 e theta2
    # theta1: o primeiro produto que temos e' o numero total de conexoes, no caso numero de elementos da camada 1 * o numero
    # de elementos na camada 2
    # theta2: estamos pegando os valores que comecam onde terminamos theta 1 e indo ate terminar(ate o final)
    # Vamos ter no final duas matrizes de peso, lembrando que essa rede que foi montada tem uma camada escondida apenas
    ###################################################################################################################

    theta1 = theta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
    #print(((input_layer_size + 1) * hidden_layer_size) + 1)
    theta2 = theta[((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
    
    theta3 = theta[((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)
    ##############################################inicio#################################################################

    m = X.shape[0]  # temos os nossos exemplos de treinamento m linhas
    J = 0  # temos que nossa funcao de custo que vai ser acumulada
    X = np.hstack((np.ones((m, 1)),X))  # atribuindo a nossa primeira coluna como valor de 1, no caso estamos atribuindo uma coluna antes de toda a matriz
    y10 = np.zeros((m, num_labels))

    #######################################valores de ativacao####################################################################
    # a1:entao agora vou fazer a primeira ativacao
    # a2:temos a nossa saida da rede
    #print(X.shape)
    #print(theta1.shape)
    # np.hstack: nesse comando estamos acrecentando uma coluna com todos os valores 1 antes da nossa matriz
    #theta2np=theta[1:21]
    a1 = sigmoid(X @ theta1.T)
    theta1nb=theta1[:,1:hidden_layer_size+1]
    theta2nb=theta2[:,1:hidden_layer_size+1]
    #print(a1.shape)
    #print(theta2.shape)# a=g(z=teta*a1), sendo que definimos a1 = x
    a1 = np.hstack((np.ones((m, 1)), a1)) 
    
    #print((theta2.T).shape)
    #print(a1.shape)# bies
    a2 = sigmoid(a1 @ theta2.T)
    a2nb=a2
    a2 = np.hstack((np.ones((m, 1)), a2))
    #print(a2.shape)
    #print((theta3.T).shape)
    a3 = sigmoid(a2 @ theta3.T)
    #print(a3.shape)
    #################################arrumando a matriz de saida/vetor##############################################
    # para o primeiro for: estamos arrumando a saida do meu y
    # para o segundo for estamos acumulando a funcao de custo
    for i in range(1, num_labels + 1):
        y10[:, i - 1][:, np.newaxis] = np.where(y == i, 1, 0)
    for j in range(num_labels):
        J = J + sum(-y10[:, j] * np.log(a3[:, j]) - (1 - y10[:, j]) * np.log(1 - a3[:, j]))
    # reg_j:Temos aqui a nossa funcao de custo regularizada

    cost = 1 / m * J
    reg_J = cost + Lambda / (2 * m) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2) + np.sum(theta3[:, 1:] ** 2))

    #################################backpropagation#####################################################
    # então eu comecei a calcular os grad1 e grad2 e ai eu preciso criar umas matrizes com o tamanho que eu quero
    # Grad1 e grad 2: estamos criando uma matriz com valores de zero depois que eu atualizar meus pesos vou armazenar os valores nele
    # newaxis: é usado para aumentar a dimensão da matriz existente em mais uma dimensão , quando usado uma vez
    grad1 = np.zeros((theta1.shape))
    grad2 = np.zeros((theta2.shape))
    grad3 = np.zeros((theta3.shape))
    #######################################Calculando as derivadas##########################################
    for i in range(m):
        # para eu calcular as derivadas entao vou precisar da ativação e dos thetas
        xi = X[i, :]  # nesse caso o meu xi e' o meu x de entrada
        a1i=a1[i,:]
        a2i=a2[i,:]
        a3i = a3[i, :]
        d3 = a3i - y10[i, :]
        d2 = np.dot(theta3.T[1:hidden_layer_size+1,:] , d3.T)
        d2_ = sigmoidGradient(np.dot(a1i, theta2.T))
        d2 *= d2_
        d1 = np.dot(d2.T, theta2nb.T)#20 # Essa função faz a mesma coisa que d1 = d2.T @ theta2.T e de forma mais rápida
        #print("Dimensão D1 : {}".format(d1.shape))
        #d1_ = sigmoidGradient(np.hstack((1, np.dot(xi, theta1.T))))
        d1_ = sigmoidGradient(np.dot(xi, theta1.T))
        #d1_ = sigmoidGradient(np.dot(xi, theta1.T))
        #print("Dimensão D1_ : {}".format(d1_.shape))
        d1 *= d1_
        # para o calculo das derivadas precisamos simplesmente dos deltas e dos as

        grad1 = grad1 + d1.T[:, np.newaxis] @ xi[:, np.newaxis].T
        grad2 = grad2 + d2.T[:, np.newaxis] @ a1i[:, np.newaxis].T
        grad3 = grad3 + d3.T[:, np.newaxis] @ a2i[:, np.newaxis].T
    # para regularizar no caso vamos precisar calcular as médias de m
    grad1 = 1 / m * grad1
    grad2 = 1 / m * grad2
    grad3 = 1 / m * grad3
    # aqui estamos fazendo a derivada regularizada
    # grad1: no caso temos que ela vai de 1 ate o fim e não contabilizamos o zero na regularizacao
    #####################################Derivadas parciais#####################################################
    # Dij=(1/m)*deltaij(I) + lambda()tetaij para j diferente de 0 , na realidade já estamos fazendo isso indo de
    # thetai[:,1:] porque não estamos regularizando a componente ou a ativacao zero
    grad1_reg = grad1 + (Lambda / m) * np.hstack((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]))
    grad2_reg = grad2 + (Lambda / m) * np.hstack((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]))
    grad3_reg = grad3 + (Lambda / m) * np.hstack((np.zeros((theta3.shape[0], 1)), theta3[:, 1:]))

    return cost, grad1, grad2, grad3, reg_J, grad1_reg, grad2_reg, grad3_reg


#######################################inicializando os meus pesos#################################################
# L_in: numero de elementos de ativacao da camada anterior/ camada anterior
# Lout: numero de elemtos de ativacao da camda de saida/ camada atual
# nesse caso essa é uma forma de inicalizacao efeciente que é demonstrada em um artigo não a muito a se tratar
def randInitializeWeights(L_in, L_out):
    epi = (6 ** 1 / 2) / (L_in + L_out) ** 1 / 2
    W = np.random.rand(L_out, L_in + 1) * (2 * epi) - epi
    return W


#########################################Gradiente descendente####################################################
# X e y são meus dados de treinamento e minhas classes que eu acabei de invocar anteriormente
# Theta: Meu vetor de pesos
# alpha: minha taxa de aprendizado
# nbr_inter: numero e iterações
# input_layer_size: numero de elementos da camada de entrada
# hidden_layer_size: numero de elementos da camada intermediaria
# num_labels: numero de classes
# m: numero de exemplos de treinamento
# J_history: vai guardar o custo em cada entrada
def gradientDescent(X, y, theta, alpha, nbr_iter, Lambda, input_layer_size, hidden_layer_size, num_labels):
    # montando então minhas matrizes de pesos theta1 e theta 2:
    theta1 = theta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = theta[((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
    theta3 = theta[((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)
    m = len(y)
    J_history = []

    for i in range(nbr_iter):
        theta = np.append(theta1.flatten(), theta2.flatten())
        theta = np.append(theta.flatten(),theta3.flatten())
        cost, grad1, grad2, grad3 = computeCost(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda)[4:]
        theta1 = theta1 - (alpha * grad1)
        theta2 = theta2 - (alpha * grad2)
        theta3 = theta3 - (alpha * grad3)
        J_history.append(cost)

    nn_paramsFinal = np.append(theta1.flatten(), theta2.flatten())
    nn_paramsFinal = np.append(nn_paramsFinal.flatten(), theta3.flatten())
    return nn_paramsFinal, J_history


#################################################funcao de predicao###########################################
# Prediction: calcula a funcao de custo no caso a saida da rede
# m:numero de exemplos de treinamneto
# X: acrecentando o bias
# argmax: no caso vai ser a minha classe de retorno
def prediction(X, theta1, theta2, theta3):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    a1 = sigmoid(X @ theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))
    a2 = sigmoid(a1 @ theta2.T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    a3 = sigmoid(a2 @ theta3.T)
    return np.argmax(a3, axis=1) + 1


# input_layer_size: numero de elementos da camada de entrada
# hidden_layer_size: numero de elementos da camada intermediaria
# num_labels: numero de classes
# o imput layer size tem tamanho 400 porque as imagens foram linearizadas e transformadas em vetores de 400
input_layer_size = 400
hidden_layer_size = 20  #Tava 25
num_labels = 10

initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size)
initial_theta3 = randInitializeWeights(hidden_layer_size, num_labels)
# No caso o initial theta vai pegar esses dois vetores iniciar e concatenar

initial_theta = np.concatenate((initial_theta1.flatten(), initial_theta2.flatten()))
initial_theta = np.concatenate((initial_theta.flatten(), initial_theta3.flatten()))

# agora calculando o theta e os custos
#Obs: estava com 1600 iteracoes
theta, J_history = gradientDescent(Xtrain, ytrain, initial_theta, 0.8, 800, 1, input_layer_size, hidden_layer_size, num_labels)
theta1 = theta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
theta2 = theta[((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
theta3 = theta[((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)
pred = prediction(Xtrain, theta1, theta2, theta3)
print("Training Set Accuracy:", sum(pred[:, np.newaxis] == ytrain)[0] / Xtrain.shape[0] * 100, "%")
pred = prediction(Xvalidation, theta1, theta2, theta3)
print("Validation Set Accuracy:", sum(pred[:, np.newaxis] == yvalidation)[0] / Xvalidation.shape[0] * 100, "%")
pred = prediction(Xtest, theta1, theta2, theta3)
print("Testing Set Accuracy:", sum(pred[:, np.newaxis] == ytest)[0] / Xtest.shape[0] * 100, "%")
print("---%sseconds--"%(time.time() - start_time))






