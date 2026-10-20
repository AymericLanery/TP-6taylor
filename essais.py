# À LA PLACE de l'habituel 'import numpy as np'
import autograd.numpy as np
from autograd import grad

#2.1
#Q1
X1=np.linspace(0,2*np.pi,50)

#2.2
#Q2
def sin_der(X):
    return np.vectorize(grad(np.sin))(X)

#2.3
#Q3
print(sin_der(X1)==np.vectorize(np.cos)(X1))

##3.
#3.1
from math import factorial
import matplotlib.pyplot as plt
def taylor1(X, derivatives):
    """
    X : le domaine (sous-entendu X[0] < 0 et X[-1] > 0)
    derivatives : derivatives[n] = f^(n)(0), dérivée n-ième de f en 0

    Retourne Y, l'approximation de Taylor sur X pour une fonction
    qui aurait ces dérivées en 0.
    """
    Y=np.zeros(len(X))
    for i in range(len(derivatives)):
        Y+=derivatives[i]*(X**i)/(factorial(i))
    return Y
sinus10 = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1]
X = np.linspace(-2 * np.pi, 2 * np.pi, 400)
#plt.plot(X,taylor1(X,sinus10))
#plt.plot(X,np.sin(X))
#plt.show()

#3.2
def taylor2(X, f, n):
    """
    X : le domaine
    f : la fonction (sur scalaire)
    n : le degré

    Retourne Y, l'approximation de Taylor de f à l'ordre n sur X.
    """
    #On commence par constuire la liste des dérivées
    derivatives=[0]*(n+1)
    g=f
    for i in range(n+1):
        derivatives[i]=g(0.0)
        g=grad(g)
    return taylor1(X,derivatives)

#tests
# exp à l'ordre 2 : 1 + x + x^2/2
assert np.allclose(taylor2(X, np.exp, 2), 1 + X + X**2 / 2)

# sinus au degré 10, à comparer avec np.sin(X)
Y = np.sin(X)
Y2 = taylor2(X, np.sin, 10)
#1.
#plt.plot(X,Y)
#plt.plot(X,Y2)
#plt.show()

#2.
def erreur(X,Y,Y2):
    return len((Y-Y2)[(Y-Y2)>10**(-3)])/len(X)
#print(erreur(X,Y,Y2))

#3.
#plt.plot(X,taylor2(X,np.cos,0))
#plt.plot(X,np.cos(X))
#plt.show()
#On constate que l'approximation est exacte seulement en 3 points

#3.3
def custom(X):
    return 2*np.sin(X)+np.cos(X/4)
Y3=custom(X)
Y4=taylor2(X,custom,14)
#plt.plot(X,Y3)
#plt.plot(X,Y4)
#plt.show()
#print(erreur(X,Y3,Y4))
