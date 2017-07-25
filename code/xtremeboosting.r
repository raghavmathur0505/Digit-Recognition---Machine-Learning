# Extreme Gradient Boosting 

# Install required packages
install.packages("readr")
library(readr)
install.packages("xgboost")
library(xgboost)

# Read the data
train1 <- digit_train
test1 <- digit_test

# Store train class values in a variable
ClassName<-train1$label

# Convert Training data into a matrix
train1<-as.matrix(train1[,-1])
train1<- matrix(as.numeric(train1),nrow(train1),ncol(train1))

# Convert Testing data into a matrix
test1<-as.matrix(test1)
test1<- matrix(as.numeric(test1),nrow(test1),ncol(test1))

# Fit a model
model<-xgboost(data = train1, label = ClassName, eta = .4 ,max.depth = 15,  nthread = 4, nround =15,
               objective = "multi:softmax",num_class=10, verbose = 2)

# Preparing predictions
digit_predictions<-data.frame(Imageid=1:nrow(test1),Label=NA)
digit_predictions[,2]<-predict(model,test1)

# View first 6 predictions
head(digit_predictions)

# Write predictions .csv file
write.csv(digit_predictions,'result_xgb-new1.csv',row.names=F)


