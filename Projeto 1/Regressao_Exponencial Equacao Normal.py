##############################################################################
##Este programa aproxima a quantidade de infectados por Coronavirus no Brasil
#nos 134 primeiros dias de infecção a partir de uma curva exponencial h(x), onde
#h(x) = theta(0)*exp(theta(1)*x), utilizando regressão linear aproximando
#log(y) por log(h), com uma taxa de aprendizado alfa.  O método utilizado é a
#Equação Normal.
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

def NormalEquation(X,y):   #Função que calcula a atualização do vetor theta a partir do Gradiente Descendente
    
    theta = np.linalg.pinv(np.dot(X.T,X)).dot(X.T).dot(y)
    Jcost = Cost(X,y,theta)
    
    return theta,Jcost

##############################################################################
####Leitura dos dados####


data = pd.read_csv("casesBrazil.csv",header=0)  #Lê o arquivo de input

##############################################################################
####Definicao das variáveis auxiliares
n =  1  #Grau do polinomio: é uma exponencial
theta = np.ones([n+1,1])   #Vetor de coeficientes inicial

##############################################################################
####Tratamento dos dados####
X,y = data["day"].values,data["cases"].values  #Extrai valores do arquivo de leitura
m = X.shape[0]  #Número de exemplos de treinamento

X = np.append(np.ones([m,1]),X.reshape(m,1),axis=1) #Inicializa a matriz com os exemplos de treinamento
Xaux = data["day"].values   #Variável auxiliar para salvar a sequência de dias (usada na plotagem mais pra frente)

y = y.reshape(m,1)  #Formata o vetor y
X = X.reshape(m,2)
z = np.log(y)   #Para aproximar log(y) (curva exponencial)

##############################################################################
####Cálculos####
new_cost = Cost(X,z,theta)  #Custo inicial
print('Custo inicial: '+ str(new_cost))

new_theta,Jcost = NormalEquation(X,z)  #Calcula a função
print('Valor de theta obtido: '+ str(new_theta))
print('Custo final: '+ str(Jcost))

##############################################################################
####Plotagem####
plt.figure(0)   #Plotagem dos dados de treinamento e aproximação
plt.scatter(Xaux,z,c='red',marker='.',label='log dos Dados de Treinamento')
plt.plot(Xaux,np.dot(X,new_theta),label='Regressão log(h(x))')    #Plota log(h(x)) = X*new_theta
plt.ylabel('Infectados')
plt.xlabel('Dias')
plt.legend()
plt.title('Gráfico Logarítmico do Ajuste por Regressão Exponencial Utilizando Equação Normal')

plt.figure(1)   #Plotagem dos dados de treinamento e aproximação
plt.scatter(Xaux,y,c='red',marker='.',label='Dados de Treinamento')
plt.plot(Xaux,np.exp(np.dot(X,new_theta)),label='Regressão Exponencial')    #Plota a exponencial de log(h(x) = h(x))
plt.ylabel('Infectados')
plt.xlabel('Dias')
plt.legend()
plt.title('Ajuste por Regressão Exponencial Utilizando Equação Normal')