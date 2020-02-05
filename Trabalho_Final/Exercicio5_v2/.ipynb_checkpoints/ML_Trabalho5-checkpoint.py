#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 22:25:13 2019

@author: andrero
"""

import seaborn as sb

from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import pandas as pd
from sklearn import svm

# Importe as bibliotecas de Pipelines e Pré-processadores
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('/Users/andrero/Google Drive/FIAP-MBA 8IA/006 - Modelos de IA e ML/Trabalhos/Trabalho5/pulsar_stars.csv', sep=',', engine='python')


#Faz a validação via crossvalidation (k-fold)
def Acuracia(clf,X,y):
    resultados = cross_val_predict(clf, X, y, cv=5)
    return metrics.accuracy_score(y,resultados)


 # Pre-processamento de dados
def remove_features(lista_features):
    for i in lista_features:
        dataset.drop(i, axis=1, inplace=True)
    return 0


# Como estamos construindo um modelo para classificar as amostras em estrelas de nêutrons ou não,
# nosso alvo será a variável "target_class" do dataframe pulsar_stars.

# Para ter certeza de que é uma variável binária, vamos usar a função countplot () do Seaborn.
sb.countplot(x='target_class',data=dataset, palette='hls')


# Visualizando as colunas
# pandas object type https://stackoverflow.com/questions/21018654/strings-in-a-dataframe-but-dtype-is-object
dataset.columns

len(dataset.columns)

dataset.dtypes

# Separa a classe dos dados
classes = dataset['target_class']
dataset.drop('target_class', axis=1, inplace=True)

# checando missing values
dataset.isnull().sum()

#Calculo do MinMaxScaler
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#X_scaled = X_std * (max - min) + min

#Calculo do StandardScaler
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#z = (x - u) / s - where u is the mean of the training samples and and s is the standard deviation of the training samples

pip_1 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC())
])

pip_2 = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('clf', svm.SVC())
])

pip_3 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='rbf'))
])

pip_4 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='poly'))
])

pip_5 = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', svm.SVC(kernel='linear'))
])


# Teste de acurácia usando o pipeline 'pip_1'
Acuracia(pip_1,dataset,classes)

Acuracia(pip_2,dataset,classes)

# # Testando Kernels 
# Kernel rbf
Acuracia(pip_3,dataset,classes)

# Kernel Polynomial
Acuracia(pip_4,dataset,classes)

# Kernel Linear
Acuracia(pip_5,dataset,classes)



# # GridSearch

from sklearn.model_selection import GridSearchCV

lista_C = [0.001, 0.01, 0.1, 1, 10,100]
lista_gamma = [0.001, 0.01, 0.1, 1, 10, 100]


parametros_grid = dict(clf__C=lista_C, clf__gamma=lista_gamma)

#Faz o tuning dos parametros testando cada combinação utilziando CrossValidation com 10 folds e analisando a acurácia
grid = GridSearchCV(pip_5, parametros_grid, cv=10, scoring='accuracy')


grid.fit(dataset,classes)

grid.cv_results_

grid.best_params_

grid.best_score_

# Métricas de Avaliação de Modelos

pip_6 = Pipeline([
('scaler',StandardScaler()),
('clf', svm.SVC(kernel='linear',C=100,gamma=0.001))
])


resultados = cross_val_predict(pip_6, dataset, classes, cv=10)

print (metrics.classification_report(classes,resultados,target_names=['0','1']))

from sklearn.metrics import confusion_matrix
#X = dataset.iloc[:,0:7].values
cm = confusion_matrix(classes, resultados)
print(cm)

#### A matriz de confusão mostra que o classificador teve os seguintes resultados:
#### Verdadeiros Positivos = 16175 (Classe Predita = Classe Esperada = Positivo para estrela de neutrons)
#### Verdadeiros Negativos = 1346 (Classe Predita = Classe Esperada = Negativo para estrela de neutrons)
#### Falsos Positivos = 293 (Classe Predita Positiva para estrela de neutrons, Classe Esperada é Negativa)
#### Falsos Negativos = 84 (Classe Predita Negativa para estrela de neutrons, Classe Esperada é Positiva )

#### Com esses resultados, podemos estimar os seguintes parâmetros:
#### Sensibilidade ou Taxa de Verdadeiros Positivos = VP / (VP + FN) = 16175 / (16175+84) = 99.48%
#### Especificidade ou Taxa de Verdadeiros Negativos = VN / (FP+VN) = 1346 / (293+1346) = 82.12%
#### Acurácia = (VP+VN) / (VP+FN+FP+VN) = 17521  / 17898 = 97.89%, que é igual ao resultado obtido com o método grid.best_score_

