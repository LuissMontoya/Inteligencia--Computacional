# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

base = pd.read_csv('diabetes_data_upload.csv')
#base = pd.read_csv('credit-g.csv')

#Separar los valores de la clase
X = base.iloc[:,0:16].values
y = base.iloc[:,16].values 


labelencoder = LabelEncoder()

X[:,0] = labelencoder.fit_transform(X[:,0])
X[:,1] = labelencoder.fit_transform(X[:,1])
X[:,2] = labelencoder.fit_transform(X[:,2])
X[:,3] = labelencoder.fit_transform(X[:,3])
X[:,4] = labelencoder.fit_transform(X[:,4])
X[:,5] = labelencoder.fit_transform(X[:,5])
X[:,6] = labelencoder.fit_transform(X[:,6])
X[:,7] = labelencoder.fit_transform(X[:,7])
X[:,8] = labelencoder.fit_transform(X[:,8])
X[:,9] = labelencoder.fit_transform(X[:,9])
X[:,10] = labelencoder.fit_transform(X[:,10])
X[:,11] = labelencoder.fit_transform(X[:,11])
X[:,12] = labelencoder.fit_transform(X[:,12])
X[:,13] = labelencoder.fit_transform(X[:,13])
X[:,14] = labelencoder.fit_transform(X[:,14])
X[:,15] = labelencoder.fit_transform(X[:,15])


#Entrenamiento
X_entrenar, X_probar, y_entrenar, y_probar = train_test_split(X, y, test_size = 0.3, random_state=0)
# tamaño de prueba = 30 % 
modelo1 = DecisionTreeClassifier(criterion='entropy')
modelo1.fit(X_entrenar, y_entrenar)
export_graphviz(modelo1, out_file = 'modelo1.dot')


# predecir
predicciones1 = modelo1.predict(X_probar)
#calcular la precisión
accuracy_score(y_probar, predicciones1)
print('salida: ',accuracy_score(y_probar, predicciones1))
#y_probar = vector de prueba
#precciones1 = vector de predicciones
#Generar la matriz de confusion
confusion1= ConfusionMatrix(modelo1)
confusion1.fit(X_entrenar, y_entrenar)
confusion1.score(X_probar, y_probar)
confusion1.poof()


#Mejorar (modelar)
modelo2 = DecisionTreeClassifier(criterion = 'entropy', min_samples_split = 100)
modelo2.fit(X_entrenar, y_entrenar)
export_graphviz(modelo2, out_file = 'modelo2.dot')

predicciones2 = modelo2.predict(X_probar)
#calcular la precisión
accuracy_score(y_probar, predicciones2)
print('salida: ',accuracy_score(y_probar, predicciones2))
#y_probar = vector de prueba
#precciones2 = vector de predicciones
#Generar la matriz de confusion
confusion2= ConfusionMatrix(modelo2)
confusion2.fit(X_entrenar, y_entrenar)
confusion2.score(X_probar, y_probar)
confusion2.poof()


#crear un 3 modelo 
modelo3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 100, min_samples_split=100)
modelo3.fit(X_entrenar, y_entrenar)
export_graphviz(modelo3, out_file = 'modelo3.dot')

predicciones3 = modelo3.predict(X_probar)
#calcular la precisión
accuracy_score(y_probar, predicciones3)
print('salida: ',accuracy_score(y_probar, predicciones3))
#y_probar = vector de prueba
#precciones3 = vector de predicciones
#Generar la matriz de confusion
confusion3= ConfusionMatrix(modelo3)
confusion3.fit(X_entrenar, y_entrenar)
confusion3.score(X_probar, y_probar)
confusion3.poof()



