from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris = datasets.load_iris()

cluster = KMeans(n_clusters =3) #poner a punto el clasificador
cluster.fit(iris.data) # realizar el entrenamiento

Prediccion = cluster.labels_
Centroides = cluster.cluster_centers_

Resultados = confusion_matrix(iris.target, Prediccion)

plt.scatter(iris.data[Prediccion == 0,0], iris.data[Prediccion ==0,3],
            c = 'green', label = 'Setosa')
plt.scatter(iris.data[Prediccion == 1,0], iris.data[Prediccion ==1,3],
            c = 'red', label = 'Versicolor')
plt.scatter(iris.data[Prediccion == 2,0], iris.data[Prediccion ==2,3],
            c = 'blue', label = 'Virgica')
plt.legend()




