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

#Cria a malha de pontos necessária para as curvas de nivel
xlist = np.arange(0,30,0.05)
ylist = np.arange(0,30,0.05)
Xpoints, Ypoints = np.meshgrid(xlist, ylist)
xylist = np.array([Xpoints.flatten(order = 'C'), Ypoints.flatten(order = 'C')]).T
zpoints = multivariateGaussian(xylist, mu, Sigma)
zpoints = zpoints.reshape(Xpoints.shape[0],Xpoints.shape[1])

#Plota as curvas de nivel com os dados da matriz X
plt.xlim(0,30)
plt.ylim(0,30)
plt.xlabel("Latência (ms)")
plt.ylabel("Taxa de transferência (mb/s)")
plt.contour(Xpoints, Ypoints, zpoints)
plt.scatter(X[:,0],X[:,1],marker="x")