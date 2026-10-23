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
    ordre=st.slider("Choisissez une valeur", 0, 10,3)
    largeur=st.slider("Choisissez la largeur de l'intervalle étudié",2,20,12)
    #Ajout pour la question 5.1:
    seuil=st.select_slider("Choisissez un seuil", options=[10**(-6),10**(-5),10**(-4),10**(-3),10**(-2)], value=10**(-3))
    #Ajout pour la question 5.2:
    afficher=st.checkbox("Afficher l'erreur")

#4.4
X=np.linspace(-largeur/2,largeur/2,400)
Y=FONCTIONS[f](X)
Y_taylor=taylor2(X,FONCTIONS[f],ordre)
fig, ax = plt.subplots()
ax.plot(X,Y, label=f,color="blue")
#la ligne suivante est mise en commentaire car on modifie à la question 6
#ax.plot(X,Y_taylor, label=f+" approché à l'ordre "+str(ordre), color="orange")
#le ax.legend et st.pyplot(fig) est placé en fin de programme

#Question
#Quand on s'éloigne de 0, et à n fixé,
#l'approximation est de moins en moins bonne
#Pour ln(1+x), quand on augment n, on alterne entre un polynôme qui tend vers - l'infini
#et un polynôme qui tend vers + l'infini

#5
#5.1
#Je m'inspire de ma fonction erreur de la partie 2
def taux_coincidence(Y,Y_taylor,seuil):
    return len((Y-Y_taylor)[(Y-Y_taylor)<=seuil])/len(X)
ax.set_title("Taux de coïncidence à "+str(seuil)+ " : "+str(taux_coincidence(Y,Y_taylor,seuil)))
#Question
#Avec mon implémentation, la comparaison avec nan est exacte
#donc cela augmente artificiellement le taux de coincidence

#5.2
if afficher:
    ax.plot(X,abs(Y-Y_taylor),label="erreur",color="green")

#6.
with st.sidebar:
    if "ordres" not in st.session_state:
        st.session_state.ordres = []
    if st.button("Ajouter cet ordre à la comparaison"):
        if ordre not in st.session_state.ordres:
            st.session_state.ordres.append(ordre)
    if st.button("Réinitialiser"):
        st.session_state.ordres=[]
        
    st.write("Ordres comparés actuellement :", st.session_state.ordres)


#tracé
for k in st.session_state.ordres:
    Y_taylor_k = taylor2(X, FONCTIONS[f], k)
    ax.plot(X, Y_taylor_k, label=f"{f} approché à l'ordre {k}")

ax.legend()
st.pyplot(fig)