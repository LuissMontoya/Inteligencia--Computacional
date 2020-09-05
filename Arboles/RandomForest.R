install.packages("randomForest")
library(randomForest)

Credito = read.csv(file.choose(), sep=",",header =T)

modelo = randomForest(class͂ ., data = Credito, ntree =500)

#Pronostico basado en los datos out of bag
modelo$predicted

#importancia de los atributos en modelo
modelo$importance

#Proporción de votos para la clasificación en cada arbol generado
modelo$votes

#arboles inducidos
modelo$forest

#Matriz de confusión basada en los datos de out of bag
modelo$confusion

plot(modelo)

#Pronostico con registro
predict(modelo, newdata = credito[154,])

#Pronostico con registro
predict(modelo, newdata = credito[437,])

