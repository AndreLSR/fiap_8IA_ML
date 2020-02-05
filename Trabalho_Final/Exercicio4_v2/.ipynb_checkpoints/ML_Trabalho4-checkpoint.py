#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:01:03 2019

@author: andrero
"""


# coding: utf-8


import pandas as pd
import seaborn as sb

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report


from sklearn.metrics import confusion_matrix

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz, plot_tree

from sklearn.preprocessing import StandardScaler

import pydotplus
import matplotlib.pyplot as plt




import os
os.environ['PATH'].split(os.pathsep)
os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'


# A primeira coisa que vamos fazer é ler o conjunto de dados usando a função read_csv() dos Pandas. 
# Colocaremos esses dados em um DataFrame do Pandas, chamado "heart":
heart = pd.read_csv('/Users/andrerodrigues/Google Drive/FIAP-MBA 8IA/006 - Modelos de IA e ML/Trabalhos/Trabalho4/heart.csv', sep = ',')
heart.head()



# Como estamos construindo um modelo para prever o grau de risco cardíaco de cada paciente , 
# nosso alvo será a variável "Target" do dataframe heart.

# Para ter certeza de que é uma variável binária, vamos usar a função countplot () do Seaborn.
sb.countplot(x='target',data=heart, palette='hls')
# Ok, agora veja que a variavel Target é binária



# # Checking for missing values
# É fácil checar missing values usando método isnull() com o método sum(), o número retornado condiz com a quantidade True 
# para o teste, ou seja, quantidade de valores nulos nas variaveis

heart.isnull().sum()

heart.info()

# Ok, então existem 303 linhas no dataframe. Nenhum composto por missing values.


# Agora temos um conjunto de dados com todas as variáveis no formato correto!
# # Validando independencia entre as variáveis
sb.heatmap(heart.corr())  



# # Agora vamos lá!!
# 1º: Separar o conjunto em variavel resposta e variaveis de treinamento
heart.head()
X = heart.iloc[:,0:-1].values
y = heart.iloc[:,-1].values

XNew = StandardScaler().fit_transform(X)
Xold = X
X = XNew

# Agora dividir em treino e teste (teste com 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)


Classif_tree = DecisionTreeClassifier()
Classif_tree.fit(X_train, y_train)
y_pred = Classif_tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))

# Exportação com o plot_tree
plot_tree(Classif_tree, filled=True, feature_names = heart.columns.values[0:-1], class_names = ['Saudavel', 'Doente'])


#Gerando a Arvore de decisão com apenas 3 nós folha
Classif_tree = DecisionTreeClassifier(max_leaf_nodes  = 3)
Classif_tree.fit(X_train, y_train)
y_pred = Classif_tree.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(Classif_tree, out_file=dot_data,
                filled=True, rounded=True,
                feature_names = heart.columns.values[0:-1],
                class_names = ['Saudável', 'Doente'],
                special_characters=True)
 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

# Vamos descobrir os maiores valores de acurácia variando a quantidade de folhas da árvore:
leaf_range = range(2, 20)
scores = []

for i in leaf_range:
    Classif_tree = DecisionTreeClassifier(max_leaf_nodes  = i)
    Classif_tree.fit(X_train, y_train)
    y_pred = Classif_tree.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print(scores)

#Plota os valores de acc. em função da quantidade de leafs:

plt.plot(leaf_range, scores)
plt.xlabel('Number of Leafs')
plt.ylabel('Testing Accuracy')

# Notamos no gráfico, que a maior acurácia (79%) é obtida com 4 e 8 folhas. Escolheremos nossa árvore com 4 folhas para atender
# o que nos foi solicitado no exercício: "...gerar a menor arvore possível com a maior taxa de acerto (podando a quantidade máxima
# de folhas e/ou altura da arvore).

Classif_tree = DecisionTreeClassifier(max_leaf_nodes  = 4)
Classif_tree.fit(X_train, y_train)
y_pred = Classif_tree.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(Classif_tree, out_file=dot_data,
                filled=True, rounded=True,
                feature_names = heart.columns.values[0:-1],
                class_names = ['Saudável', 'Doente'],
                special_characters=True)
 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())





