
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import pandas as pd
import scipy.optimize as spopt
import time
start_time=time.time()
#mat3:Tem informações sobre as notas
#mat4:tem informações sobre os atributos dos filmes e parametros dos usuarios
mat3 = loadmat("dado3.mat")

#Y: no caso a nossa resposta, nota do usuario para o filme
#R:variavel de ligacao que indica se o filme foi votado ou nao
#X:vetor/matriz de parametros para cada filme
#Theta: Matriz de parametros para cada usuario
Y = mat3["Y"]
R = mat3["R"]


R=R[:,:]
Y=Y[:,:]
#aqui estamos calculando a média das notas que o filme recebeu, no caso so entra na media os filmes que receberam nota
print("Average rating for movie 1 (Toy Story):",np.sum(Y[0,:]*R[0,:])/np.sum(R[0,:]),"/5")




#aqui estamos plotando um color map pra saber como estam distribuido as notas de cada filme com cada usuario
plt.figure(figsize=(8,16))
plt.imshow(Y)
plt.xlabel("Users")
plt.ylabel("Movies")
plt.show() #acrescentei isso aqui

def  cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    #X:vetor/matriz de parametros para cada filme
    #Theta: Matriz de parametros para cada usuario
    #Os parâmetros X( parâmetros de conteúdo do filme) e 
    #Theta(parâmetros orientados ao usuário) são inseridos como um único bloco params.
    #prediction: nota predita 
    #err: erro
    #J: funcao de custo da filtragem colaborativa
    #reg_x,regTheta: funcao de custo regularizada
    #X_grad: calculo do gradiente X
    #Theta_grad: calculo do gradiente Theta
    #grad: funcao gradiente
    #reg_X_grad,reg_Theta_grad,reg_grad: mesmo calculo diferente e a somatoria lambda X,theta
    global grad
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)
    
    predictions =  X @ Theta.T
    err = (predictions - Y)
    J = 1/2 * np.sum((err**2) * R)
    

    reg_X =  Lambda/2 * np.sum(Theta**2)
    reg_Theta = Lambda/2 *np.sum(X**2)
    reg_J = J + reg_X + reg_Theta
    

    X_grad = err*R @ Theta
    Theta_grad = (err*R).T @ X
    grad = np.append(X_grad.flatten(),Theta_grad.flatten())
    

    reg_X_grad = X_grad + Lambda*X
    reg_Theta_grad = Theta_grad + Lambda*Theta
    reg_grad = np.append(reg_X_grad.flatten(),reg_Theta_grad.flatten())
    
    return J, reg_grad,reg_J, grad


#################################################################################
#movieList = open("movie_ids.txt","r").read().split("\n")[:-1]
#carregando uma lista de filmes 
movieList = open("dado4.txt","r",encoding="utf-8").read().split("\n")[:]

my_ratings = np.zeros((1682,1))

my_ratings[0] = 4 
my_ratings[13] = 2
my_ratings[9] = 1
my_ratings[3] = 0
my_ratings[4] = 3
my_ratings[100] = 5
my_ratings[44] = 3
my_ratings[250] = 5
my_ratings[300] = 3
my_ratings[88] = 5
my_ratings[440] = 2
my_ratings[180] = 5

print("Notas do novo usuário:\n")
for i in range(len(my_ratings)):
    if my_ratings[i]>0:
        print("Nota",int(my_ratings[i]),"para o índice",movieList[i])
        
        
      
        
def normalizeRatings(Y, R):
    #Normalizando as notas, normalizacao pela media 
    #Ynorm: vetor Y normalizado
    #Ymean: vetor Y medio
    m,n = Y.shape[0], Y.shape[1]
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros((m,n))
    
    for i in range(m):
        Ymean[i] = np.sum(Y[i,:])/np.count_nonzero(R[i,:])
        Ynorm[i,R[i,:]==1] = Y[i,R[i,:]==1] - Ymean[i]
        
    return Ynorm, Ymean




def FuncaoCusto(initial_parameters):
    global Ynorm, R, num_users, num_movies, num_features, Lambda
    return cofiCostFunc(initial_parameters,Ynorm,R,num_users,num_movies,num_features,Lambda)[0]
 



def cofi_costfunc_grad(Y, R, num_users, num_movies, num_features, lambda_c, params):
  
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users,num_features)
    
    X_grad = np.dot(((np.dot(X, Theta.T) - Y) * R), Theta) + lambda_c * X
    Tgsub = (np.dot(X, Theta.T) - Y) * R
    Theta_grad = np.dot(Tgsub.T, X) + lambda_c * Theta
    Jgrad = np.concatenate((X_grad.ravel(), Theta_grad.ravel()))

    return Jgrad





def compute_grad_sp(initial_parameters):
    global Ynorm, R, num_users, num_movies, num_features, Lambda
    j_grad = cofi_costfunc_grad(Ynorm, R, num_users, num_movies, num_features, Lambda, params=initial_parameters)

    return j_grad





#############################################################################################
#colocando as notas na minha matriz Y
#Arrumando minha matriz R colocando os valores sendo 0(nota nao dada) ou 1(nota atribuida)
Y = np.hstack((my_ratings,Y))
R =np.hstack((my_ratings!=0,R))
Ynorm, Ymean = normalizeRatings(Y, R)

#NUm_users: numero de usuarios(correspondente a nossas colunas)
#num_movies: numero de filmes(corresponde ao numero de linhas)
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10


#X: matriz X iniciada aleatoriamente
#Theta:Matriz theta iniciada aleatoriamente
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.append(X.flatten(),Theta.flatten())
Lambda = 40



options={ 'maxiter': 190000,'disp': True, 'gtol': 1e-15}
#disp:True para imprimir mensagens de convergência.
#maxiter: numero maximo de iteracoes
#gtol: tolerancia de parada

#J_history: utulizando a funcao minimize(method='CG')                                                                                                                                          
J_history = spopt.minimize(FuncaoCusto, initial_parameters, method='CG', jac=compute_grad_sp,options=options)
J_history2=J_history.x 

plt.plot(J_history2)
plt.xlabel("initial_parameters")
plt.ylabel("$J(\Theta)$")
plt.title("Função de Custo usando Gradiente Conjugado")
plt.show() #acrescentei isso aqui


Xnew = grad[:num_movies*num_features].reshape(num_movies,num_features)
Thetanew = grad[num_movies*num_features:].reshape(num_users,num_features)
alpha=0.0001
X= X - (alpha*Xnew)
Theta=Theta - (alpha*Thetanew)
#p: probabilidade x*theta.T
p = X@ Theta.T
#plt.figure(figsize=(16,16))
#plt.imshow(p)
#plt.show()
#no final preciso voltar com o velor medio que acabei subtraindo 
my_predictions = p[:,0][:,np.newaxis] + Ymean


#ordenacao dos indices, para serem recomendados 
df = pd.DataFrame(np.hstack((my_predictions,np.array(movieList)[:,np.newaxis])))
df.sort_values(by=[0],ascending=False,inplace=True)
df.reset_index(drop=True,inplace=True)


print("Melhores recomendações para você:\n")
for i in range(10):
    print("Nota predita",round(float(df[0][i]),1)," para o índice",df[1][i])
    
print("---%sseconds--"%(time.time() - start_time))
print("o valor de alpha",alpha)
print("saves modificados")
    

