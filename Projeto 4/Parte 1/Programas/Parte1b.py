import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

################################################################################################
#Funções
def estimateGaussian(X):
    
    m = X.shape[0]
    
    sum_ = np.sum(X,axis=0)
    mu = 1/m *sum_
    
    sigma = 1/m * ((X - mu).T)@(X - mu)    
    return mu,sigma

def multivariateGaussian(X, mu, Sigma):
       
    k = len(mu)
    
    X = X - mu.T
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(Sigma)**0.5))* np.exp(-0.5* np.sum(X @ np.linalg.pinv(Sigma) * X,axis=1))
    return p
################################################################################################
#Leitura dos dados
mat = loadmat("dado1.mat")
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"]

#Cálculo da média e da matriz de covariâncias da matriz X
mu, Sigma = estimateGaussian(X)

#Retorna o valor z da gaussiana nos pontos pertencentes à matriz X
p = multivariateGaussian(X, mu, Sigma)