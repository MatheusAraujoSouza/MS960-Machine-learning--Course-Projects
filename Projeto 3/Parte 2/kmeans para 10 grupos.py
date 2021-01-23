
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
    
    
    for i in range(num_iters):
        centroids = computeCentroids(X,idx,K)
        idx = findClosestCentroids(X,centroids)
        
 
    
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
centroids = KMeansInitCentroids(Zn,10)

idx = findClosestCentroids(Zn,centroids)
plotKmeans(Zn,centroids,idx,10,200)
primeiraM = np.zeros((100,1024), dtype=np.float64)
segundaM = np.zeros((100,1024), dtype=np.float64)
terceiraM = np.zeros((100,1024), dtype=np.float64)
quatroM= np.zeros((100,1024), dtype=np.float64)
cincoM= np.zeros((100,1024), dtype=np.float64)
sixM= np.zeros((100,1024), dtype=np.float64)
seteM= np.zeros((100,1024), dtype=np.float64)
oitoM= np.zeros((100,1024), dtype=np.float64)
noveM= np.zeros((100,1024), dtype=np.float64)
dezM= np.zeros((100,1024), dtype=np.float64)
primeiranozero= np.zeros((100,1024), dtype=np.float64)
segundanozero= np.zeros((100,1024), dtype=np.float64)
terceiranozero = np.zeros((100,1024), dtype=np.float64)
quatronozero = np.zeros((100,1024), dtype=np.float64)
quintanozero= np.zeros((100,1024), dtype=np.float64)
sextanozero= np.zeros((100,1024), dtype=np.float64)
setimanozero= np.zeros((100,1024), dtype=np.float64)
oitavanozero= np.zeros((100,1024), dtype=np.float64)
nonanozero= np.zeros((100,1024), dtype=np.float64)
decimanozero= np.zeros((100,1024), dtype=np.float64)

#primeiranozero = np.zeros((100,2), dtype=np.float64)
#segundanozero = np.zeros((100,2), dtype=np.float64)
#terceiranozero= np.zeros((100,2), dtype=np.float64)
#########separando os grupos em matrizes######################################
for i in range(100):
    if idx[i]==1:
        primeiraM[i,:]=X_norm[i,:]
        
    elif idx[i]==2:
        segundaM[i,:]=X_norm[i,:]
    elif idx[i]==3:
        terceiraM[i,:]=X_norm[i,:]
    elif idx[i]==4:
        quatroM[i,:]=X_norm[i,:]
    elif idx[i]==5:
        cincoM[i,:]=X_norm[i,:]
    elif idx[i]==6:
        sixM[i,:]=X_norm[i,:]
    elif idx[i]==7:
        seteM[i,:]=X_norm[i,:]
    elif idx[i]==8:
        oitoM[i,:]=X_norm[i,:]
    elif idx[i]==9:
        noveM[i,:]=X_norm[i,:]
    else:
        dezM[i,:]=X_norm[i,:]


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
contzer=0
for k in range(100):
    if quatroM[k,k] != 0 : 
        quatronozero[contzer,:]=quatroM[k,:]
        contzer+=1   
contzer=0
for k in range(100):
    if cincoM[k,k] != 0 : 
        quintanozero[contzer,:]=cincoM[k,:]
        contzer+=1    
contzer=0
for k in range(100):
    if sixM[k,k] != 0 : 
        sextanozero[contzer,:]=sixM[k,:]
        contzer+=1  
contzer=0
for k in range(100):
    if seteM[k,k] != 0 : 
        setimanozero[contzer,:]=seteM[k,:]
        contzer+=1 
contzer=0
for k in range(100):
    if oitoM[k,k] != 0 : 
        oitavanozero[contzer,:]=oitoM[k,:]
        contzer+=1  
contzer=0
for k in range(100):
    if oitoM[k,k] != 0 : 
        oitavanozero[contzer,:]=oitoM[k,:]
        contzer+=1        
contzer=0
for k in range(100):
    if noveM[k,k] != 0 : 
        nonanozero[contzer,:]=noveM[k,:]
        contzer+=1  
contzer=0
for k in range(100):
    if dezM[k,k] != 0 : 
        decimanozero[contzer,:]=dezM[k,:]
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
        
Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(quatronozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1
              
Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(quintanozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1
                  
Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(sextanozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1
                       
Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(setimanozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1    
                       
Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(oitavanozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1    
Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(nonanozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1                 
Con=0
fig, axis = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axis[i, j].imshow(decimanozero[Con, :].reshape(32, 32, order="F"), cmap="gray")
        axis[i, j].axis("off")
        Con+=1           
        