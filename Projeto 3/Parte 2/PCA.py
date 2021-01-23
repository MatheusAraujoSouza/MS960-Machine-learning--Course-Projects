# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:36:22 2020

@author: 55199
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from numpy.linalg import svd

################################## Parte I ###################################
mat = loadmat("dado3.mat")
X=mat.get('X')




def plot_de_imagens(X):
    fig, axis = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axis[i, j].imshow(X[np.random.randint(0, 5001), :].reshape(32, 32, order="F"), cmap="gray")
            axis[i, j].axis("off")



def X_normalizado(X):
    mu=np.mean(X,axis=0)
    sigma=np.std(X,axis=0)
    
    X_norm=(X-mu)/sigma
    return X_norm,sigma,mu

##########################Analise de componentes principais####################
#precisamos agora verificar o numero de componentes principais que sao suficientes
def PCA(X):
    m,n=X.shape[0],X.shape[1]
    sigma=1/m*X.T @ X
    U,S,V = svd(sigma)
    return U,S,V



#calculando a U reduziada 

def projectData(X,U,K):
    m = X.shape[0]
    U_reduced = U[:,:K]
    Z = np.zeros((m,K))
    
    for i in range(m):
        for j in range(K):
            Z[i,j] = X[i,:] @ U_reduced[:,j]
            
    return Z

def recoverData(Z,U,K):
    m,n = Z.shape[0],U.shape[1]
    X_rec = np.zeros((m,n))
    U_reduced = U[:,:K]
    
    for i in range(m):
        X_rec[i,:] = Z[i,:] @ U_reduced.T
        
    return X_rec





norm_resultado=X_normalizado(X)
X_norm=norm_resultado[0]
sigma_norm=norm_resultado[1]

U,S,V =PCA(X_norm)
#Os autovalores são armazenados dentro da matriz U 
#Os autovetores de nossa matriz são as colunas de V 
#Os elementos da diagonal da matriz S são os valores singulares são as raizes 
#quadradas dos autovalores não nulos de A
mu=norm_resultado[2]
plot_de_imagens(X_norm)

    


################################# Parte II ###################################





Con=0
fig, axis = plt.subplots(6, 6, figsize=(8, 8))
for i in range(6):
    for j in range(6):
        axis[i, j].imshow(U[:, Con].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1


################################### Parte III ################################

K=100


Z = projectData(X_norm,U,K)




X_rec = recoverData(Z,U,K)



Con=0
fig, axis = plt.subplots(6, 6, figsize=(8, 8))
for i in range(6):
    for j in range(6):
        axis[i, j].imshow(X_rec[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1
            
Con=0
fig, axis = plt.subplots(6, 6, figsize=(10, 10))
for i in range(6):
    for j in range(6):
        axis[i, j].imshow(X[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1        
            





