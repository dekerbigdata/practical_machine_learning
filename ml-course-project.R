setwd("~/Projects/practical_machine_learning")

library(caret)
library(randomForest)
library(gbm)
library(rpart)
library(MASS)


# load an clean data

train_set <- read.csv('data/pml-training.csv')
tests_set <- read.csv('data/pml-testing.csv')

train_set <- train_set[,colSums(is.na(train_set)) == 0]
tests_set <- tests_set[,colSums(is.na(tests_set)) == 0]

train_set <- train_set[,-c(1:7)]
tests_set <- tests_set[,-c(1:7)]

nzvc <- nearZeroVar(train_set, saveMetrics = TRUE)
train_set <- train_set[, nzvc$nzv==FALSE]

# create cross-validation

set.seed(1)
index <- createDataPartition(y=train_set$classe, p=0.6, list=FALSE)
train_subset <- train_set[index, ] 
cross_subset <- train_set[-index, ]

# model
library(parallel)
library(doParallel)
registerDoParallel(makeCluster(detectCores()))

m_rf <- train(classe ~ ., method = 'rf', data = train_subset, verbose=FALSE)   
m_gbm <- train(classe ~ ., method = 'gbm', data = train_subset, verbose=FALSE)
m_rpart <- train(classe ~ ., method = 'rpart', data = train_subset, verbose=FALSE)
m_lda <- train(classe ~ ., method = 'lda', data = train_subset, verbose=FALSE)

# validation - selection
pred_rf <- predict(m_rf, cross_subset); confusionMatrix(pred_rf, cross_subset$classe)$overall
pred_gbm <- predict(m_gbm, cross_subset); confusionMatrix(pred_gbm, cross_subset$classe)$overall
pred_lda <- predict(m_lda, cross_subset); confusionMatrix(pred_lda, cross_subset$classe)$overall

imp_var <- varImp(m_rf)
plot(imp_var, main='Importance of variables')

# results - testing

predict_test <- predict(m_rf, tests_set, type="raw")
predict_test

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(predict_test)
