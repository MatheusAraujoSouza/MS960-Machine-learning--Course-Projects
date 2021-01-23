
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

def U_reds(X,U,K):
    m = X.shape[0]
    U_reduced = U[:,:K]
    Z = np.zeros((m,K))
    
    for i in range(m):
        for j in range(K):
            Z[i,j] = X[i,:] @ U_reduced[:,j]
            
    return U_reduced


def recoverData(Z,U,K):
    m,n = Z.shape[0],U.shape[1]
    X_rec = np.zeros((m,n))
    U_reduced = U[:,:K]
    
    for i in range(m):
        X_rec[i,:] = Z[i,:] @ U_reduced.T
        
    return X_rec

############################funções para o kmeans ############################
def findClosestCentroids(X,centroids):
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0],1))
    temp = np.zeros((centroids.shape[0],1))
    
    for i in range(X.shape[0]):
        for j in range(K):
            dist = X[i,:]-centroids[j,:]
            length = np.sum(dist**2)
            temp[j] = length
        idx[i] = np.argmin(temp)+1
    return idx

def computeCentroids(X,idx,K):
    m,n = X.shape[0],X.shape[1]
    centroids = np.zeros((K,n))
    count = np.zeros((K,1))
    
    for i in range(m):
        index = int((idx[i]-1)[0])
        centroids[index,:] += X[i,:]
        count[index]+=1
        
    return centroids/count


def plotKmeans(X,centroids,idx,K,num_iters):
    
    m,n = X.shape[0],X.shape[1]
    
    fig,ax = plt.subplots(nrows=num_iters,ncols=1,figsize = (6,36))
    
    for i in range(num_iters):
        color = "rgb"
        for k in range(1,K+1):
            grp = (idx==k).reshape(m,1)
            ax[i].scatter(X[grp[:,0],0],X[grp[:,0],1],c = color[k-1],s=15)
            
        ax[i].scatter(centroids[:,0],centroids[:,1],s=120,marker="x",c="black",linewidth=3)
        title = "Número de iterações"+str(i)
        ax[i].set_title(title)
        
        centroids = computeCentroids(X,idx,K)
        
        idx = findClosestCentroids(X,centroids)
        
    plt.tight_layout()
    
def KMeansInitCentroids(X,K):
    m,n = X.shape[0],X.shape[1]
    centroids = np.zeros((K,n))
    
    for i in range(K):
        centroids[i] = X[np.random.randint(0,m+1),:]
        
    return centroids
    
    
    


norm_resultado=X_normalizado(X)
X_norm=norm_resultado[0]
sigma_norm=norm_resultado[1]








######################Criando os grupos no k-means#############################


#casos_inciais = np.zeros((10,1024), dtype=np.float64)
casos_iniciais=X_norm[0:100,:]
#agora comprimindo os dados 

Un,Sn,Vn=PCA(casos_iniciais)

Zn=projectData(casos_iniciais,Un,2)
centroids = KMeansInitCentroids(Zn,3)

idx = findClosestCentroids(Zn,centroids)
plotKmeans(Zn,centroids,idx,3,20)
primeiraM = np.zeros((100,1024), dtype=np.float64)
segundaM = np.zeros((100,1024), dtype=np.float64)
terceiraM = np.zeros((100,1024), dtype=np.float64)
primeiranozero= np.zeros((100,1024), dtype=np.float64)
segundanozero= np.zeros((100,1024), dtype=np.float64)
terceiranozero = np.zeros((100,1024), dtype=np.float64)
#primeiranozero = np.zeros((100,2), dtype=np.float64)
#segundanozero = np.zeros((100,2), dtype=np.float64)
#terceiranozero= np.zeros((100,2), dtype=np.float64)
#########separando os grupos em matrizes######################################
for i in range(100):
    if idx[i]==1:
        primeiraM[i,:]=X_norm[i,:]
        
    elif idx[i]==2:
        segundaM[i,:]=X_norm[i,:]
    else:
        terceiraM[i,:]=X_norm[i,:]


contzer=0
for k in range(100):
    if primeiraM[k,k] != 0 : 
        primeiranozero[contzer,:]=primeiraM[k,:]
        contzer+=1

contzer=0
for k in range(100):
    if segundaM[k,k] != 0 : 
        segundanozero[contzer,:]=segundaM[k,:]
        contzer+=1
contzer=0
for k in range(100):
    if terceiraM[k,k] != 0 : 
        terceiranozero[contzer,:]=terceiraM[k,:]
        contzer+=1                          
#plots das imagagnes que ele classificou como mesmo grupo 

#plots dos grupos



Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(primeiranozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1
        
Con=0       
fig, axis = plt.subplots(4, 4, figsize=(15, 15))        
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(segundanozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1
Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(terceiranozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1