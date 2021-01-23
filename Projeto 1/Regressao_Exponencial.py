##############################################################################
#Este programa aproxima a quantidade de infectados por Coronavirus no Brasil
#nos 134 primeiros dias de infecção a partir de uma curva exponencial h(x), onde
#h(x) = theta(0)*exp(theta(1)*x), utilizando regressão linear aproximando
#log(y) por log(h), com uma taxa de aprendizado alfa. O método utilizado é o
#Gradiente Descendente.
#
#Autores: Gabriel Borin Macedo  RA: 197201
#         Matheus Araujo Souza  RA: 184145
#         Raul Augusto Teixeira RA: 205177
#
##############################################################################
####Bibliotecas####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
##############################################################################

####Funções####
def Cost(X,y,theta):    #Função para o cálculo do custo da Regressão
    m = len(y)  #Número de exemplos de treinamento
    h = X.dot(theta)    #Função de regressão calculada até agora. Resultará na aproximação log(h) ~ log(y)
    J = (1/(2*m))*(np.sum((h-y)**2))#Função de custo
    return J

def gradientDescent(X,y,theta,alfa,num_iter):   #Função que calcula a atualização do vetor theta a partir do Gradiente Descendente
    
    J_his = []  #Inicializa vetor de custos a cada iteração
    m = len(y)  #Número de exemplos de treinamento
    for i in range(num_iter):   #Faça num_iter vezes:
        h = X.dot(theta)    #Aproxima os valores de y pela regressão linear
        theta = theta - (alfa/m)*(X.T.dot(h-y)) #Atualiza os valores de theta
        J_his.append(Cost(X,y,theta))   #Calcula o novo custo e atualiza o vetor de custos
    
    return theta,J_his

##############################################################################
####Leitura dos dados####


data = pd.read_csv("casesBrazil.csv",header=0)  #Lê o arquivo de input

##############################################################################
####Definicao das variáveis auxiliares
n =  1  #Grau do polinomio: é uma exponencial
alfa = 0.001  #Taxa de aprendizado
theta = np.ones([n+1,1])   #Vetor de coeficientes inicial
num_iter = 15000 #Número de iterações do gradiente descendente

##############################################################################
####Tratamento dos dados####
X,y = data["day"].values,data["cases"].values  #Extrai valores do arquivo de leitura
m = X.shape[0]  #Número de exemplos de treinamento

X = np.append(np.ones([m,1]),X.reshape(m,1),axis=1) #Inicializa a matriz com os exemplos de treinamento
Xaux = data["day"].values   #Variável auxiliar para salvar a sequência de dias (usada na plotagem mais pra frente)
X[:,1] = (X[:,1] - min(X[:,1]))/(max(X[:,1]) - min(X[:,1])) #Normaliza a segunda coluna da matriz X
  
y = y.reshape(m,1)  #Formata o vetor y
z = np.log(y)   #Para aproximar log(y) (curva exponencial)

##############################################################################
####Cálculos####
new_cost = Cost(X,z,theta)  #Custo inicial
print('Custo inicial: ' + str(new_cost))

new_theta,J_his = gradientDescent(X,z,theta,alfa,num_iter)  #Calcula a função
print('Valor de theta obtido: ' + str(new_theta))
print('Custo final: ' + str(Cost(X,z,new_theta)))

##############################################################################
####Plotagem####
plt.figure(0)   #Plotagem dos dados de treinamento e aproximação
plt.scatter(Xaux,z,c='red',marker='.',label='log dos Dados de Treinamento')
plt.plot(Xaux,np.dot(X,new_theta),label='Regressão log(h(x))')    #Plota log(h(x)) = X*new_theta
plt.ylabel('Infectados')
plt.xlabel('Dias')
plt.legend()
plt.title('Gráfico Logarítmico do Ajuste por Regressão Exponencial, alfa = ' + str(alfa))

plt.figure(1)   #Plotagem dos dados de treinamento e aproximação
plt.scatter(Xaux,y,c='red',marker='.',label='Dados de Treinamento')
plt.plot(Xaux,np.exp(np.dot(X,new_theta)),label='Regressão Exponencial')    #Plota a exponencial de log(h(x) = h(x))
plt.ylabel('Infectados')
plt.xlabel('Dias')
plt.legend()
plt.title('Ajuste por Regressão Exponencial, alfa = ' + str(alfa))

plt.figure(2)   #Plota o custo J(theta) a cada iteração
plt.plot(range(len(J_his)),J_his,label = 'Função de Custo Logarítmico')
plt.xlabel('Número de iterações')
plt.ylabel('Custo')
plt.legend()
plt.title('Função Custo por Iteração, alfa = ' + str(alfa))