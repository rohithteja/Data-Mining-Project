library(caret)
library(rlist)
#install.packages("MLmetrics")
library(MLmetrics)
library(Amelia)
set.seed(69)
data <- read.csv("Data_Cortex_Nuclear_edit.csv",na.strings = c(" ", "NA"), header = T, stringsAsFactors =T)
summary(data)

sum(is.na(data))
micedata[!complete.cases(data),]


#fixing missing values
head(data)
data <- data[,2:82]

# removing mouse ID
data_factor <- data[,78:81]
data_numeric <- data[,1:77]
data_numeric$pS6_N <- NULL
head(data$pS6_N)
sum(is.na(data$pS6_N))

#imputing the missing values
data_numeric_amelia <- amelia(x = data_numeric)
data_numeric_imputed <- data_numeric_amelia$imputations$imp3
sum(is.na(data_numeric_imputed))
sum(is.na(data_factor))

#building model with numeric values and the categorical variables
data_model <- cbind(data_numeric_imputed,data_factor)

#removing unnecessary categorical variables
data_model <- data_model[,-c(77,78,79)]

#splitting the data into train and test
index <- createDataPartition(data_model$class, p=0.30, list=FALSE)
train <- data_model[-index,]
test <- data_model[index,]


#CASE 1 - model building without PCA
# random forest model with 10 fold cross validation
fit.rf <- train(class~., data=train, method="rf", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.rf$results)
predictions_rf <- predict(fit.rf, test)
confusionMatrix(predictions_rf, test$class)
#variable importance of the model
varImp(fit.rf$finalModel)
accuracy.rf <- Accuracy(predictions_rf,test[,77])


#svm
fit.svm <- train(class~., data=train, method="svmRadial", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.svm$results)
predictions_svm <- predict(fit.svm, test)
confusionMatrix(predictions_svm, test$class)
accuracy.svm <- Accuracy(predictions_svm,test[,77])

#knn
fit.knn <- train(class~., data=train, method="knn", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.knn$results)
predictions_knn <- predict(fit.knn, test)
confusionMatrix(predictions_knn, test$class)
accuracy.knn <- Accuracy(predictions_knn,test[,77])

#nn
fit.nn <- train(class~., data=train, method="nnet", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.nn$results)
predictions_nn <- predict(fit.nn, test)
confusionMatrix(predictions_nn, test$class)
accuracy.nn <- Accuracy(predictions_nn,test[,77])

#nb
fit.nb <- train(class~., data=train, method="nb", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.nb$results)
predictions_nb <- predict(fit.nb, test)
confusionMatrix(predictions_nb, test$class)
accuracy.nb <- Accuracy(predictions_nb,test[,77])


# summarize accuracy of models
results <- resamples(list( KNN=fit.knn, RandomForest=fit.rf,SVM=fit.svm,NeuralNetworks=fit.nn,NaiveBayes=fit.nb))
summary(results)

accuracy_Case1 <- data.frame( KNN=accuracy.knn, RandomForest=accuracy.rf,SVM=accuracy.svm,NeuralNetworks=accuracy.nn,NaiveBayes=accuracy.nb)
rownames(accuracy_Case1) <- "TestAccuracy"
accuracy_Case1 <- t(accuracy_Case1)
accuracy_Case1



#CASE 2 - PCA
pca <- prcomp(data_numeric_imputed, scale=TRUE)
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100,1)
barplot(pca.var.per, main = "Principal Component Analysis",xlab = "Principal components",ylab = "Variance")


data_model_pca <- pca$x[,1:10]
data_model_pca <- cbind(data_model_pca,data_factor)
data_model_pca <- data_model_pca[,-c(11,12,13)]

index_pca <- createDataPartition(data_model_pca$class, p=0.30, list=FALSE)
train_pca <- data_model_pca[-index_pca,]
test_pca <- data_model_pca[index_pca,]

#model building withPCA
# rf.pca
fit.rf.pca <- train(class~., data=train_pca, method="rf", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.rf.pca$results)
predictions_rf.pca <- predict(fit.rf.pca, test_pca)
confusionMatrix(predictions, test_pca$class)
varImp(fit.rf.pca$finalModel)
accuracy.rf.pca <- Accuracy(predictions_rf.pca,test_pca[,11])

#svm.pca
fit.svm.pca <- train(class~., data=train_pca, method="svmRadial", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.svm.pca$results)
predictions_svm.pca <- predict(fit.svm.pca, test_pca)
confusionMatrix(predictions_svm.pca, test_pca$class)
accuracy.svm.pca <- Accuracy(predictions_svm.pca,test_pca[,11])

#knn
fit.knn.pca <- train(class~., data=train_pca, method="knn", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.knn.pca$results)
predictions_knn.pca <- predict(fit.knn.pca, test_pca)
confusionMatrix(predictions_knn.pca, test_pca$class)
accuracy.knn.pca <- Accuracy(predictions_knn.pca,test_pca[,11])

#nn
fit.nn.pca <- train(class~., data=train_pca, method="nnet", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.nn.pca$results)
predictions_nn.pca <- predict(fit.nn.pca, test_pca)
confusionMatrix(predictions_nn.pca, test_pca$class)
accuracy.nn.pca <- Accuracy(predictions_nn.pca,test_pca[,11])

#nb
fit.nb.pca <- train(class~., data=train_pca, method="nb", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.nb.pca$results)
predictions_nb.pca <- predict(fit.nb.pca, test_pca)
confusionMatrix(predictions_nb.pca, test_pca$class)
accuracy.nb.pca <- Accuracy(predictions_nb.pca,test_pca[,11])

# summarize accuracy of models
results.pca <- resamples(list( KNN.PCA=fit.knn.pca, RandomForest.PCA=fit.rf.pca,SVM.PCA=fit.svm.pca,NeuralNetworks.PCA=fit.nn.pca,NaiveBayes.PCA=fit.nb))
summary(results.pca)

accuracy_Case2 <- data.frame( KNN.PCA=accuracy.knn.pca, RandomForest.PCA=accuracy.rf.pca,SVM.PCA=accuracy.svm.pca,NeuralNetworks.PCA=accuracy.nn.pca,NaiveBayes.PCA=accuracy.nb.pca)
rownames(accuracy_Case2) <- "TestAccuracy"
accuracy_Case2 <- t(accuracy_Case2)
accuracy_Case2



# CASE 3 - TOP 5 features
# varImp(fit.rf$finalModel) gives the list of features with their 
# importance values. Top 5 features are selected
train5 <- train[,c("SOD1_N","pERK_N","CaNA_N","DYRK1A_N","pPKCG_N","class")]
test5 <- test[,c("SOD1_N","pERK_N","CaNA_N","DYRK1A_N","pPKCG_N","class")]

#model building with 5 features
# rf
fit.rf5 <- train(class~., data=train5, method="rf", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.rf5$results)
predictions_rf5 <- predict(fit.rf5, test5)
confusionMatrix(predictions.rf5, test5$class)
varImp(fit.rf5$finalModel)
varimp <- varImp(fit.rf5$finalModel)
accuracy.rf5 <- Accuracy(predictions_rf5,test5[,6])


#svm
fit.svm5 <- train(class~., data=train5, method="svmRadial", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.svm5$results)
predictions_svm5 <- predict(fit.svm5, test5)
confusionMatrix(predictions_svm5, test5$class)
accuracy.svm5 <- Accuracy(predictions_svm5,test5[,6])

#knn
fit.knn5 <- train(class~., data=train5, method="knn", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.knn5$results)
predictions_knn5 <- predict(fit.knn5, test5)
confusionMatrix(predictions_knn5, test5$class)
accuracy.knn5 <- Accuracy(predictions_knn5,test5[,6])

#nn
fit.nn5 <- train(class~., data=train5, method="nnet", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.nn5$results)
predictions_nn5 <- predict(fit.nn5, test5)
confusionMatrix(predictions_nn5, test5$class)
accuracy.nn5 <- Accuracy(predictions_nn5,test5[,6])

#nb
fit.nb5 <- train(class~., data=train5, method="nb", metric="Accuracy",trControl=trainControl(method="cv", number=10))
print(fit.nb5$results)
predictions_nb5 <- predict(fit.nb5, test5)
confusionMatrix(predictions_nb5, test5$class)
accuracy.nb5 <- Accuracy(predictions_nb5,test5[,6])


# summarize accuracy of models
results5 <- resamples(list( KNN=fit.knn5, RandomForest=fit.rf5,SVM=fit.svm5,NeuralNetworks=fit.nn5,NaiveBayes=fit.nb5))
summary(results5)

accuracy_Case3 <- data.frame( KNN=accuracy.knn5, RandomForest=accuracy.rf5,SVM=accuracy.svm5,NeuralNetworks=accuracy.nn5,NaiveBayes=accuracy.nb5)
rownames(accuracy_Case3) <- "TestAccuracy"
accuracy_Case3 <- t(accuracy_Case3)
accuracy_Case3	