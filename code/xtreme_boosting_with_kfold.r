training$=factor(cancer$vesinv)
# Computing Training accuracy, AOC and Confusion Matrix for Extreme 
# Gradient Boosting using 10-fold Cross-Validation

# Install and load necessary packages
install.packages("xgboost")
library(xgboost)

install.packages("ROCR")
library(ROCR)

install.packages("pROC")
library(pROC)

install.packages("caret")
library(caret)

# Read the data
data1 <- digit_train

# Specify no. of folds
nrFolds <- 10

# Specify class column index
ClassCol<-as.integer(1) 
data1[,ClassCol]<-as.numeric(data1[,ClassCol])

# Store Classes in variable
Class<-data1[,ClassCol]

# Generate array containing fold-number for each sample (row)
folds <- rep_len(1:nrFolds, nrow(data1))

# Actual cross validation
for(k in 1:nrFolds) {
  cat("Fold ",k,"\n")
  
  # Splitting  of the data into train and test
  fold <- which(folds == k)
  
  # Creating train data
  data.train <- data1[-fold,]
  ClassName<- data.train$label
  f <- as.formula(paste(ClassName," ~."))
  data.train<-as.matrix(data.train[,-1])
  data.train<- matrix(as.numeric(data.train),nrow(data.train),ncol(data.train))
  
  # Creating test data
  data.test <- data1[fold,]
  test_data <- data.test
  Class1<-data.test[,ClassCol]
  data.test<-as.matrix(data.test)
  data.test<- matrix(as.numeric(data.test),nrow(data.test),ncol(data.test))
  
  # train and test the gradient boosting model with data.train and data.test
  model<-xgboost(data = data.train, label = ClassName, eta = .4 ,max.depth = 15,  nthread = 4, nround =15,
                 objective = "multi:softmax",num_class=10)
  
  pred<-predict(model,data.test) 
  data<-data.frame('predict'=pred, 'actual'=Class1)
  
  # Compute train accuracy
  count<-nrow(data[data$predict==data$actual,])
  total<-nrow(data.test)
  count
  avg = (count*100)/total
  avg =format(round(avg, 2), nsmall = 2)
  avg  #example of how to output
  method<-"Gradient Boosting"
  accuracy<-avg
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  # Write data to file
  sink("accuracy.txt", append=TRUE)
  cat("Fold ",k,"\nAccuracy = ", accuracy)
  cat("\n----------------------------------------------------------------------------------------------\n")
  sink()
  
  # Compute AOC
  value<-multiclass.roc(Class1, pred, level=base::levels(as.factor(ClassName)), percent=TRUE)
  
  # Write data to file
  sink("aoc.txt", append=TRUE)
  cat("Fold ",k,"\n")
  sink()
  capture.output(value, file = "aoc.txt", append=TRUE)
  sink("aoc.txt", append=TRUE)
  cat("----------------------------------------------------------------------------------------------\n")
  sink()
  
  # Compute Confusion Matrix
  table(pred)
  sort.list(pred)
  pred<-round(pred)
  test_label <-data.test[,1]
  tables<- confusionMatrix(data=pred, test_label)
  print(tables)
  
  # Calculate Precision and Recall
  m<-tables$byClass
  
  precision_vals<-c(m[1,5], m[2,5], m[3,5], m[4,5], m[5,5], m[6,5], m[7,5], m[8,5], m[9,5], m[10,5])
  avg_precision<-mean(precision_vals)
  
  recall_vals<-c(m[1,6], m[2,6], m[3,6], m[4,6], m[5,6], m[6,6], m[7,6], m[8,6], m[9,6], m[10,6])
  avg_recall<-mean(recall_vals)
  
  # Write data to file
  sink("confusion_matrix.txt", append=TRUE)
  cat("Fold ",k,"\n")
  sink()
  capture.output(tables, file = "confusion_matrix.txt", append=TRUE)
  sink("confusion_matrix.txt", append=TRUE)
  cat("\nPrecision = ", avg_precision)
  cat("\nRecall = ", avg_recall)
  cat("\n----------------------------------------------------------------------------------------------\n")
  sink()
}
#6.48 to 7.13