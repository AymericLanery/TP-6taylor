#1. Découverte de Streamlit

#1.1 Première application

import streamlit as st

st.title("Approximation de Taylor")
st.write("Bienvenue dans votre outil de visualisation.")
st.header("On va implémenter la formule de Taylor et l'appliquer")

#1.2
st.latex(r"P_{n}(x)=\sum_{k=0}^{n}\frac{f^{(k)}(0)}{k!}x^k")

#1.3
#valeur = st.slider("Choisissez une valeur", 0, 10)
#st.write("Vous avez choisi :", valeur)
#Le script n'est exécuté qu'une seule fois

#2. #3. voir essais.py
#4.
import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

#4.1
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

def custom(X):
    return 2*np.sin(X)+np.cos(X/4)

#4.2
FONCTIONS={"sinus":np.sin,"cosinus":np.cos,"exp":np.exp,"ln":lambda x: np.log(1+x)}

#4.3
with st.sidebar:
    f=st.selectbox("Choisissez une fonction",FONCTIONS)
    st.slider("Choisissez une valeur", 0, 10,3)
    largeur=st.slider("Choisissez la largeur de l'intervalle étudié",2,20,12)

#4.4
X=np.arange(-largeur/2,largeur/2,400)
Y=f(X)
Y_taylor=taylor2(X,f,10)