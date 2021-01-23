import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Leitura dos dados
mat = loadmat("dado1.mat")
X = mat["X"]
Xval = mat["Xval"]
yval = mat["yval"]

#Plotagem dos dados
plt.scatter(X[:,0],X[:,1],marker="x")
plt.xlim(0,30)
plt.ylim(0,30)
plt.xlabel("Latência (ms)")
plt.ylabel("Taxa de transferência (mb/s)")