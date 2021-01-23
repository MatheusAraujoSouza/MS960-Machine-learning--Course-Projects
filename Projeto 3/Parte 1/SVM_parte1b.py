import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

mat = loadmat("dado2.mat")
X = mat["X"]
y = mat["y"]
Xval = mat["Xval"]
yval = mat["yval"]


m,n = X.shape[0],X.shape[1]
pos,neg = (y==1).reshape(m,1),(y==0).reshape(m,1)

classifier1 = SVC(C = 1, kernel="rbf")
classifier50 = SVC(C = 50, kernel="rbf")
classifier100 = SVC(C = 100, kernel="rbf")

classifier1.fit(X,np.ravel(y))  #Treinos
classifier50.fit(X,np.ravel(y))
classifier100.fit(X,np.ravel(y))

X_1,X_2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,1].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))

plt.figure(figsize=(8,6))
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50, label = "pos")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="y",marker="o",s=50, label = "neg")

contour1 = plt.contour(X_1,X_2,classifier1.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")

contour50 = plt.contour(X_1,X_2,classifier50.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="purple")

contour100 = plt.contour(X_1,X_2,classifier100.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="green")

plt.title("Fronteiras de decisao para C=1 (azul), C=50 (roxo) e C=100 (verde)")

plt.xlim(-0.62,0.32)
plt.ylim(-0.68,0.63)