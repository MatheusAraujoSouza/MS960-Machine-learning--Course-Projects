import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#ypredict = classifier1.predict(Xval)

mat = loadmat("dado2.mat")
X = mat["X"]
y = mat["y"]
Xval = mat["Xval"]
yval = mat["yval"]

num_exemplos = yval.shape[0]
best_accuracy = -1

for C in np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):
    
    for sigma in np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]):

        #Training:
        gamma = 1/(2*(sigma)**2)
        classifier = SVC(gamma = gamma, C = C, kernel="rbf")
        classifier.fit(X,np.ravel(y))  #Treinos
        
        #Validation:
        pred = classifier.predict(Xval)
        
        #accuracy = (sum(pred==yval.all()))/(num_exemplos)*100
        accuracy = accuracy_score(yval,pred)
        
        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_sigma = sigma
            best_C = C
        
        print("Training Set Accuracy for C =", C ,", Sigma =", sigma, ":", round(accuracy,2)*100,"%")

print("Best sigma:",best_sigma)
print("Best C:",best_C)
print("Best accuracy:",round(best_accuracy,2)*100,"%")

#Plotting the best obtained parameters:
    
classifier = SVC(gamma = 1/(2*(best_sigma)**2), C = best_C, kernel="rbf")
classifier.fit(X,np.ravel(y))  #Treinos

m,n = Xval.shape[0],Xval.shape[1]
pos,neg = (yval==1).reshape(m,1),(yval==0).reshape(m,1)

X_1,X_2 = np.meshgrid(np.linspace(Xval[:,0].min(),Xval[:,1].max(),num=100),np.linspace(Xval[:,1].min(),Xval[:,1].max(),num=100))

plt.figure(figsize=(8,6))
plt.scatter(Xval[pos[:,0],0],Xval[pos[:,0],1],c="r",marker="+",s=50, label = "pos")
plt.scatter(Xval[neg[:,0],0],Xval[neg[:,0],1],c="y",marker="o",s=50, label = "neg")

contour = plt.contour(X_1,X_2,classifier.predict(np.array([X_1.ravel(),X_2.ravel()]).T).reshape(X_1.shape),1,colors="b")

plt.title('Fronteiras de decisao para C = '+str(best_C)+' e sigma = '+str(best_sigma))

plt.xlim(-0.82,0.52)
plt.ylim(-0.88,0.83)