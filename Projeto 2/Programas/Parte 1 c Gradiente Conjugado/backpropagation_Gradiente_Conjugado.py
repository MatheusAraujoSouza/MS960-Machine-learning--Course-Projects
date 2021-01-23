# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:13:46 2020

@author: 55199
"""

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
from scipy.optimize import minimize
# A funcao load mat estamos fazendo um intercambio entre os dados do matlab
# entao ele vai abrir o arquivo ja linearizado ponto mat e assim
# Ja vou pedir minha matriz de dados tanto para a variavel x quanto para a variavel y


mat = loadmat("ex3data1.mat")
Xaux = mat["X"]
yaux = mat["y"]

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


def computeCost(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda,num_layers, num_training):
    #############################################comentarios#####################################################
    # Theta 1 e Theta 2
    # No caso no inicio vamos criar duas matrizes de peso  theta1 e theta2
    # theta1: o primeiro produto que temos e' o numero total de conexoes, no caso numero de elementos da camada 1 * o numero
    # de elementos na camada 2
    # theta2: estamos pegando os valores que comecam onde terminamos theta 1 e indo ate terminar(ate o final)
    # Vamos ter no final duas matrizes de peso, lembrando que essa rede que foi montada tem uma camada escondida apenas
    ###################################################################################################################
    theta_guard = np.zeros((num_layers-1,hidden_layer_size,hidden_layer_size+1), dtype=np.float64)
    theta_guard_nb = np.zeros((num_layers-1,hidden_layer_size,hidden_layer_size), dtype=np.float64)
    Grad_Guard=np.zeros((num_layers-1,hidden_layer_size,hidden_layer_size+1), dtype=np.float64)
    Grad_Guard_reg=np.zeros((num_layers-1,hidden_layer_size,hidden_layer_size+1), dtype=np.float64)
    a_guard = np.zeros((num_layers,num_training,hidden_layer_size+1), dtype=np.float64)
    ansig = np.zeros((num_layers,num_training,hidden_layer_size), dtype=np.float64)
    a_int_delta=np.zeros((num_layers,hidden_layer_size+1,1), dtype=np.float64)
    delta_int=np.zeros((num_layers-1,hidden_layer_size,1), dtype=np.float64)
    ##############################################inicio#################################################################



    theta1 = theta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
    for i in range(num_layers -1 ):
        
        if i == 0:
            theta_guard[i,:,:] = theta[((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
        else:
            theta_guard[i,:,:] = theta[((hidden_layer_size + 1) * hidden_layer_size)*(i)+((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)*(i+1)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
    theta3 = theta[((hidden_layer_size + 1) * hidden_layer_size)*(num_layers-1)+((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)
        
    
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
    ##########################################Loop para os ativarores##############################################################
    for k in range(num_layers-1):
        theta_guard_nb[k,:,:]=theta_guard[k,:,1:hidden_layer_size+1]
        
    
    a1 = sigmoid(X @ theta1.T)                             
    a1 = np.hstack((np.ones((m, 1)), a1))
    a_guard[0,:,:]=a1
    # a=g(z=teta*a1), sendo que definimos a1 = x
    
    
    
    
    
    a2 = sigmoid(a1 @ theta_guard[0,:,:].T)
    a2 = np.hstack((np.ones((m, 1)), a2))
    
    
    ###################com isso meu a1 esta feito########
    ansig[0,:,:]=sigmoid(a1 @ theta_guard[0,:,:].T)
    a_guard[1,:,:]=np.hstack((np.ones((m, 1)), ansig[0,:,:]))
    for i in range(num_layers-2):  
        ansig[i+1,:,:]=sigmoid(a_guard[i+1,:,:] @ theta_guard[i+1,:,:].T)
        a_guard[i+2,:,:]=np.hstack((np.ones((m, 1)), ansig[i+1,:,:]))
    a3 = sigmoid(a_guard[num_layers-1,:,:] @ theta3.T)
  
        
       
   
   
    
    #a3 = sigmoid(a2 @ theta3.T)
    #ate aqui mantemos os loops que temos que travar, so precisamos voltar para 
    #acertar o tamanho do para ativadores pequenos 
    ##########################################loop para os thetas##################################################
   
    #################################arrumando a matriz de saida/vetor##############################################
    # para o primeiro for: estamos arrumando a saida do meu y
    # para o segundo for estamos acumulando a funcao de custo
    for i in range(1, num_labels + 1):
        y10[:, i - 1][:, np.newaxis] = np.where(y == i, 1, 0)
    for j in range(num_labels):
        J = J + sum(-y10[:, j] * np.log(a3[:, j]) - (1 - y10[:, j]) * np.log(1 - a3[:, j]))
    # reg_j:Temos aqui a nossa funcao de custo regularizada

    cost = 1 / m * J
    reg_int=0
    for i in range(num_layers-1):
        reg_int=reg_int+np.sum(theta_guard[i,:, 1:] ** 2)
    reg_J = cost + Lambda / (2 * m) * (np.sum(theta1[:, 1:] ** 2) + reg_int + np.sum(theta3[:, 1:] ** 2))

    #################################backpropagation#####################################################
    # então eu comecei a calcular os grad1 e grad2 e ai eu preciso criar umas matrizes com o tamanho que eu quero
    # Grad1 e grad 2: estamos criando uma matriz com valores de zero depois que eu atualizar meus pesos vou armazenar os valores nele
    # newaxis: é usado para aumentar a dimensão da matriz existente em mais uma dimensão , quando usado uma vez
    #aqui novamente vamos criar uma matriz de grads para armazenar esses valores
    grad1 = np.zeros((theta1.shape))
    grad3 = np.zeros((theta3.shape))
    #######################################Calculando as deltas e derivadas##########################################
    for i in range(m):
        # para eu calcular as derivadas entao vou precisar da ativação e dos thetas
        xi = X[i, :]  # nesse caso o meu xi e' o meu x de entrada
        for j in range(num_layers):
            a_int_delta[j,:,0] = a_guard[j,i,:]
        

        a1i=a1[i,:]
        a2i=a2[i,:]
        a3i = a3[i, :]
        d3 = a3i - y10[i, :]
        
        
        
        
        d2 = np.dot(theta3.T[1:hidden_layer_size+1,:] , d3.T)
        d2_ = sigmoidGradient(np.dot(a_int_delta[num_layers-2,:,0], theta_guard[num_layers-2,:,:].T))
        d2 *= d2_
        delta_int[0,:,0]=d2  
        
        
        
        
        
        for l in range(num_layers-2):
            delta_int[l+1,:,0] = np.dot(theta_guard_nb[l+1,:,:].T , delta_int[l,:,0].T)
            delta_int_aux= sigmoidGradient(np.dot(a_int_delta[num_layers-2-(l+1),:,0], theta_guard[num_layers-2-(l+1),:,:].T))
            delta_int[l+1,:,0] *= delta_int_aux
            
        d1 = np.dot(delta_int[num_layers-2,:,0].T, theta_guard_nb[0,:,:].T)
        d1_ = sigmoidGradient(np.dot(xi, theta1.T))
        d1 *= d1_
        
        
        grad1 = grad1 + d1.T[:, np.newaxis] @ xi[:, np.newaxis].T
        for b in range(num_layers-1):
            Grad_Guard[b,:,:] = Grad_Guard[b,:,:] + delta_int[b,:,0].T[:, np.newaxis] @ a_int_delta[b,:, np.newaxis].T
        grad3 = grad3 + d3.T[:, np.newaxis] @ a_int_delta[num_layers-1,:, np.newaxis].T
    # para regularizar no caso vamos precisar calcular as médias de m
    grad1 = 1 / m * grad1
    Grad_Guard[:,:,:]=1/m*Grad_Guard[:,:,:]
    grad3 = 1 / m * grad3
    # aqui estamos fazendo a derivada regularizada
    # grad1: no caso temos que ela vai de 1 ate o fim e não contabilizamos o zero na regularizacao
    #####################################Derivadas parciais#####################################################
    # Dij=(1/m)*deltaij(I) + lambda()tetaij para j diferente de 0 , na realidade já estamos fazendo isso indo de
    # thetai[:,1:] porque não estamos regularizando a componente ou a ativacao zero
    grad1_reg = grad1 + (Lambda / m) * np.hstack((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]))
    for j in range(num_layers-1):
        Grad_Guard_reg[j,:,:] = Grad_Guard[j,:,:] + (Lambda / m) * np.hstack((np.zeros((theta_guard[j,:,:].shape[0], 1)), theta_guard[j,:, 1:]))
    grad3_reg = grad3 + (Lambda / m) * np.hstack((np.zeros((theta3.shape[0], 1)), theta3[:, 1:]))

    return cost, grad1, Grad_Guard, grad3, reg_J, grad1_reg, Grad_Guard_reg, grad3_reg


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
def gradientDescent(X, y, theta, alpha, nbr_iter, Lambda, input_layer_size, hidden_layer_size, num_labels,num_layers,num_training):
    # montando então minhas matrizes de pesos theta1 e theta 2:
        theta_guard_1 = np.zeros((num_layers-1,hidden_layer_size,hidden_layer_size+1), dtype=np.float64)
      
        
      
        
      
        theta1 = theta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
        for i in range(num_layers -1 ):
            if i == 0:
                theta_guard_1[i,:,:] = theta[((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
            else:
                theta_guard_1[i,:,:] = theta[((hidden_layer_size + 1) * hidden_layer_size)*(i)+((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)*(i+1)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
        theta3 = theta[((hidden_layer_size + 1) * hidden_layer_size)*(num_layers-1)+((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)
        
        
#agora vamnos criar um for dentro desse laço para armazenar os thetas e fazer as comparações     
   
    
        m = len(y)
        J_history = []

        for i in range(nbr_iter):
            
            theta = np.append(theta1.flatten(), theta_guard_1[0,:,:].flatten())
            for j in range(num_layers-2):
                theta = np.append(theta.flatten(), theta_guard_1[j+1,:,:].flatten())
            theta = np.append(theta.flatten(),theta3.flatten())
            
            cost, grad1, Grad_Guard, grad3 = computeCost(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda,num_layers)[4:]
           
            
            theta1 = theta1 - (alpha * grad1)
            for g in range(num_layers-1):
                theta_guard_1[g,:,:] = theta_guard_1[g,:,:]  - (alpha * Grad_Guard[g,:,:])
            theta3 = theta3 - (alpha * grad3)
            J_history.append(cost)

        
        
        
        
        
        nn_paramsFinal = np.append(theta1.flatten(), theta_guard_1[0,:,:].flatten())
        for j in range(num_layers-2):
            nn_paramsFinal = np.append(nn_paramsFinal.flatten(), theta_guard_1[j+1,:,:].flatten())
        nn_paramsFinal = np.append(nn_paramsFinal.flatten(),theta3.flatten())
        
        
        
        
        return nn_paramsFinal, J_history


#################################################funcao de predicao###########################################
# Prediction: calcula a funcao de custo no caso a saida da rede
# m:numero de exemplos de treinamneto
# X: acrecentando o bias
# argmax: no caso vai ser a minha classe de retorno
def prediction(X, theta1, theta_guard, theta3, num_training):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    a_guard = np.zeros((num_layers-1,num_training,hidden_layer_size+1), dtype=np.float64)

    a1 = sigmoid(X @ theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))
    aux2 = sigmoid(a1 @ theta_guard[0,:,:].T)
    a_guard[0,:,:] = np.hstack((np.ones((m, 1)), aux2))
    for i in range(num_layers-2):
        aux4 = sigmoid(a_guard[i,:,:] @ theta_guard[i+1,:,:].T)
        a_guard[i+1,:,:] = np.hstack((np.ones((m, 1)), aux4))
    a3 = sigmoid(a_guard[num_layers-2,:,:] @ theta3.T)
    
    
    
    
    return np.argmax(a3, axis=1) + 1
######################################################################################################

def FuncaoCusto(theta, X, y, Lambda ,input_layer_size, hidden_layer_size,num_labels, num_layers, num_training):
    return computeCost(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda,num_layers, num_training)[4]  #Retorna o custo regularizado

######################################################################################################

# input_layer_size: numero de elementos da camada de entrada
# hidden_layer_size: numero de elementos da camada intermediaria
# num_labels: numero de classes
# o imput layer size tem tamanho 400 porque as imagens foram linearizadas e transformadas em vetores de 400
input_layer_size = 400  #Tamanho da imagem input
hidden_layer_size = 5 #Numero de neuronios das camadas internas
num_labels = 10 #Numero de classes
num_layers=3    #Numero de camadas internas
Lambda = 1;
num_iter = 100  #Numero de iteracoes maximo do Gradiente Conjugado

num_training = 10  #Numero de exemplos totais 


X = Xaux[0,:].reshape(1,400)
y = yaux[0].reshape(1,1)
j = 1


for i in range(Xaux.shape[0]):
    if(yaux[i] == j):
        aux = i
        X = np.append(X,Xaux[aux,:].reshape(1,400), axis = 0)
        y = np.append(y,yaux[aux].reshape(1,1), axis = 0) 
        if(j == 9):
            break
        j += 1
    
print("Exemplos selecionados!")






###################################escolha da quantidade de camdas escondidas#################
#declarando os dois thetas iniciais
initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2= randInitializeWeights(hidden_layer_size, hidden_layer_size)
initial_theta3 = randInitializeWeights(hidden_layer_size, num_labels)
theta_guard_3 = np.zeros((num_layers,hidden_layer_size,hidden_layer_size+1), dtype=np.float64)


#######################################loop para colocar as camadas internas#######################

#if num_layers > 1:
#    
#    theta_temp1 = randInitializeWeights(hidden_layer_size, hidden_layer_size)
#    initial_theta = np.concatenate((initial_theta1.flatten(), theta_temp1.flatten()))
#    for i in range(num_layers-1): 
#      
#        initial_theta = np.concatenate((initial_theta.flatten(), theta_temp1.flatten()))
#
#    initial_theta = np.concatenate((initial_theta.flatten(), initial_theta3.flatten()))
#else:
#    initial_theta = np.concatenate((initial_theta1.flatten(), initial_theta3.flatten()))
#############################################################################################

theta_temp1 = randInitializeWeights(hidden_layer_size, hidden_layer_size)
initial_theta = np.concatenate((initial_theta1.flatten(), theta_temp1.flatten()))
for i in range(num_layers - 2):
    initial_theta = np.concatenate((initial_theta.flatten(), theta_temp1.flatten()))
initial_theta = np.concatenate((initial_theta.flatten(), initial_theta3.flatten()))    
    


# agora calculando o theta e os custos
#theta, J_history = gradientDescent(X, y, initial_theta, 3.0, num_iter, Lambda, input_layer_size, hidden_layer_size, num_labels, num_layers)


theta = minimize(FuncaoCusto, initial_theta, method='CG', options={'maxiter': num_iter, 'disp': True, 'gtol': 1e-3}, args=(X, y, Lambda ,input_layer_size, hidden_layer_size,num_labels,num_layers,num_training))
theta = theta.x

####################################loop para organizar a saida das variaveis###################





theta1 = theta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
for i in range(num_layers -1 ):    
    if i == 0:
        theta_guard_3[i,:,:] = theta[((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
    else:
        theta_guard_3[i,:,:] = theta[((hidden_layer_size + 1) * hidden_layer_size)*(i)+((input_layer_size + 1) * hidden_layer_size):((hidden_layer_size + 1) * hidden_layer_size)*(i+1)+((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, hidden_layer_size + 1)
theta3 = theta[((hidden_layer_size + 1) * hidden_layer_size)*(num_layers-1)+((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)
pred = prediction(X, theta1, theta_guard_3, theta3, num_training)
print("Training Set Accuracy:", sum(pred[:, np.newaxis] == y)[0] / num_training * 100, "%")
print("---%sseconds--"%(time.time() - start_time))