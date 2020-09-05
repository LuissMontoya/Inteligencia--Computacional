install.packages("randomForest")

library(randomForest)
library(randomForest)

Credito = read.csv(file.choose(), sep=",",header =T)
modelo = randomForest(class ~ .,data= Credito, ntree =500)

#modelo$predicted
plot(modelo)

