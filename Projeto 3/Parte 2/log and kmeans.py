

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
 ###################################funcoes regressaolog#######################
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
    
###############################################################################    


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
primeiranozeroZn= np.zeros((100,2), dtype=np.float64)
segundanozeroZn= np.zeros((100,2), dtype=np.float64)
terceiranozeroZn = np.zeros((100,2), dtype=np.float64)
segundanozeroZn_new= np.zeros((100,2), dtype=np.float64)
#primeiranozero = np.zeros((100,2), dtype=np.float64)
#segundanozero = np.zeros((100,2), dtype=np.float64)
#terceiranozero= np.zeros((100,2), dtype=np.float64)
#########separando os grupos em matrizes######################################
for i in range(100):
    if idx[i]==1:
        primeiraM[i,:]=X_norm[i,:]
        
        
    elif idx[i]==2:
        segundaM[i,:]=X_norm[i,:]
        segundanozeroZn[i,:]=Zn[i,:]
    else:
        terceiraM[i,:]=X_norm[i,:]
        terceiranozeroZn[i,:]=Zn[i,:]


contzer=0
for k in range(100):
    if primeiraM[k,k] != 0 : 
        primeiranozero[contzer,:]=primeiraM[k,:]
        primeiranozeroZn[contzer,:]=Zn[k,:]
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
        
contazero=0
for g in range(100):
    if segundanozeroZn[g,0] !=0 :
        segundanozeroZn_new[contazero,:]=segundanozeroZn[g,:]
        contazero+=1





        
M =np.zeros((10,2), dtype=np.float64)
M[0,:]=primeiranozeroZn[0,:]
M[1:10,:]=segundanozeroZn_new[1:10,:] 





        
        
m = M.shape[0]
n = M.shape[1]


Z__L = np.append(np.ones([m,1]),M,axis=1)
Pguard =np.zeros((10,10), dtype=np.float64)
y =np.zeros((10,1), dtype=np.float64)
for i in range(10):
    y[i,0]=i+1
        
Yguard=y     
 
theta = np.zeros([n+1,1])
cost = computeCost(Z__L,y,theta)  
print(cost)
nbr_iter = 100000
alpha = 0.1





for g in range(0,10):
    for i in range(1,11):  
   
        for j in range(0,10):
            if y[j,0] == i:
                y[j,0] = 1
            else:
                y[j,0] = 0

        new_theta,J_history = gradientDescent(Z__L,y,theta,alpha,nbr_iter)
        new_cost = computeCost(Z__L,y,new_theta)
        prob = prediction(Z__L[g,:],new_theta)
        #aqui vamos voltar os valores de y
        for s in range(10):
            y[s,0]=s+1
        Pguard[g,i-1]=prob[0]
      


    