#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:26:17 2019

@author: andrero
"""
import pandas as pd
import seaborn as sb
from pandas import DataFrame as df

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



import os
os.environ['PATH'].split(os.pathsep)
os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'


# A primeira coisa que vamos fazer é ler o conjunto de dados usando a função read_csv() dos Pandas. 
# Colocaremos esses dados em um DataFrame do Pandas, chamado "dataset", e nomearemos cada uma das colunas.

dataset = pd.read_csv('/Users/andrero/Google Drive/FIAP-MBA 8IA/006 - Modelos de IA e ML/Trabalhos/Trabalho6/mushrooms.csv')
dataset.head()


# # Pre-processamento de dados
def remove_features(lista_features):
    for i in lista_features:
        dataset_int.drop(i, axis=1, inplace=True)
    return 0


# Como estamos construindo um modelo para classificar o cogumelo, nosso alvo será a variável "class" do dataframe dataset.
# Para ter certeza de que é uma variável binária, vamos usar a função countplot () do Seaborn.
sb.countplot(x='class',data=dataset, palette='hls')

# Ok, agora veja que a variavel class é binária

# Checking for missing values
# É fácil checar missing values usando método isnull() com o método sum(), o número retornado 
# condiz com a quantidade True para o teste, ou seja, quantidade de valores nulos nas variaveis

dataset.isnull().sum()

dataset.info()

dataset.columns

len(dataset.columns)

dataset.dtypes

# Vemos que o dataset foi todo construído com dtype = object. Precisamos encontrar um modo de transformá-lo
# em variávels não-categóricas. Para isso vamos usar o LabelEncoder do SciKitLearn

def enc_features(lista_features):
    enc = LabelEncoder()
    dataset_int = df(dataset)
    for i in lista_features:
        inteiros = enc.fit_transform(dataset[i])
        j = i + '_int'
        dataset_int[j] = inteiros
    return dataset_int

dataset_int = enc_features(dataset.columns)
        
dataset_int.dtypes

obj_columns = dataset_int.select_dtypes(['object']).columns
obj_columns
remove_features(obj_columns)

dataset_int.dtypes


# # Validando independencia entre as variáveis

sb.heatmap(dataset_int.corr())  

# Vemos que a variável veil-type_int não agrega nenhuma informação e, por isso, vamos remove-la:
dataset_int.drop('veil-type_int', axis=1, inplace=True)

sb.heatmap(dataset_int.corr()) 

# Agora temos um conjunto de dados com todas as variáveis no formato correto!


# # Agora vamos lá!!
# 1º: Separar o conjunto em variavel resposta e variaveis de treinamento



X = dataset_int.iloc[:,1:].values
y = dataset_int.iloc[:,0].values


# Agora dividir em treino e teste (teste com 30%)


X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)


#classificador Naive Bayes Gaussiano
classificador = GaussianNB()

classificador.fit(X_train, y_train)
#Em caso de datasets muitos granges é possível utilizar a função partial_fit
#classificador.partial_fit(X_train, y_train)
 
y_pred = classificador.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))

print(cm)