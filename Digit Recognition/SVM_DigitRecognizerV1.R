setwd("C:/Users/abhik/Desktop/Work Folder/R Projects/Digit Recognition")

############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel

#####################################################################################

# 1. Business Understanding: 

#develop a model using Support Vector Machine which should correctly classify the 
#handwritten digits based on the pixel values given as features.

#####################################################################################

# 2. Data Understanding: 


# Load required Librbaries: 
library(kernlab)
library(readr)
library(caret)
library(caTools)
library(e1071)
library(dplyr)
library(tidyr)
library(doParallel)

train_data <- read_csv("mnist_train.csv", col_names = F)
test_data <- read_csv("mnist_test.csv", col_names = F)

dim(train_data)
dim(test_data)

# This dataset contains one row for each of the 60000 training instances, 
# and one column for each of the 784 pixels in a 28 x 28 image. The data as 
# downloaded doesn't have column labels, but are arranged as "row 1 column 1, 
# row 1 column 2, row 1 column 3." and so on). 

renameColumns <- function(df){
  colnames(df) <- paste0("Y", 0:(ncol(df)-1))
  colnames(df)[which(names(df) == "Y0")] <- "DIGIT"
  return(df)
}

train_data <- renameColumns(train_data)
colnames(train_data)
test_data <- renameColumns(test_data)
colnames(test_data)

head(train_data)
head(test_data)


#####################################################################################

#3. Data Preparation:

# Changing output variable "DIGIT" to factor type 
train_data$DIGIT <- as.factor(train_data$DIGIT)
test_data$DIGIT <- as.factor(test_data$DIGIT)

# Checking missing value
sum(sapply(train_data, function(x) sum(is.na(x)))) # No missing values
sum(sapply(test_data, function(x) sum(is.na(x)))) # No missing values

sum(duplicated(train_data)) # no duplicate rows
sum(duplicated(test_data)) # no duplicate rows

# Sampling the data to make the computation faster
set.seed(100)
train_subset.indices = sample(1:nrow(train_data), 0.15*nrow(train_data))
train = train_data[train_subset.indices, ]
test_subset.indices = sample(1:nrow(test_data), 0.15*nrow(test_data))
test <- test_data[test_subset.indices, ]

# Scaling data 
max(train[ ,2:ncol(train)]) # max pixel value is 255, lets use this to scale data
train[ , 2:ncol(train)] <- train[ , 2:ncol(train)]/255
#test <- cbind(test_data[,1],test_data[ , 2:ncol(test_data)]/255)
test[ , 2:ncol(test)] <- test[ , 2:ncol(test)]/255

#####################################################################################

# 4. Model Building

# Setup Parallel Processing
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

#--------------------------------------------------------------------
# 4.1 Using Linear Kernel
#--------------------------------------------------------------------

Model_linear <- ksvm(DIGIT~ ., data = train, scaled = FALSE, kernel = "vanilladot")
print(Model_linear) 
Eval_linear<- predict(Model_linear, test)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$DIGIT)

# Using Hyperparameter Tuning and Crossvalidation
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid <- expand.grid(C = c(0, 0.1, 0.25, 0.5, 0.75, 1, 1.25))
set.seed(3233)
svm_Linear <- train(DIGIT ~., data = train, 
                    method = "svmLinear",trControl=trctrl,
                    tuneGrid = grid,
                    metric="Accuracy",
                    tuneLength = 10)
print(svm_Linear)
# Support Vector Machines with Linear Kernel 
# 
# 9000 samples
# 784 predictor
# 10 classes: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 
# 
# No pre-processing
# Resampling: Cross-Validated (3 fold) 
# Summary of sample sizes: 5998, 6000, 6002 
# Resampling results across tuning parameters:
#   
#   C     Accuracy   Kappa    
# 0.00        NaN        NaN
# 0.10  0.9087791  0.8986173
# 0.25  0.9087791  0.8986173
# 0.50  0.9087791  0.8986173
# 0.75  0.9087791  0.8986173
# 1.00  0.9087791  0.8986173
# 1.25  0.9087791  0.8986173
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was C = 0.1.
plot(svm_Linear)

# Prediction for Testset
Eval_linear<- predict(svm_Linear, test)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$DIGIT)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1   2   3   4   5   6   7   8   9
# 0 144   0   1   1   0   0   3   1   1   4
# 1   0 157   3   0   0   2   1   1   1   3
# 2   0   0 143   0   1   0   3   2   0   0
# 3   0   0   4 136   0   6   0   1   9   0
# 4   0   0   1   1 131   1   5   1   2   8
# 5   0   1   0   4   0 102   3   0   6   0
# 6   1   1   1   0   0   3 144   0   0   0
# 7   0   0   0   1   0   0   0 136   0   4
# 8   0   1   4   6   1   0   1   0 133   1
# 9   0   0   0   0   3   0   0   6   1 158
# 
# Overall Statistics
# 
# Accuracy : 0.9227         
# 95% CI : (0.908, 0.9357)
# No Information Rate : 0.1187         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.914          
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
# Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity           0.99310   0.9812  0.91083  0.91275  0.96324  0.89474   0.9000  0.91892  0.86928   0.8876
# Specificity           0.99188   0.9918  0.99553  0.98520  0.98607  0.98990   0.9955  0.99630  0.98961   0.9924
# Pos Pred Value        0.92903   0.9345  0.95973  0.87179  0.87333  0.87931   0.9600  0.96454  0.90476   0.9405
# Neg Pred Value        0.99926   0.9977  0.98964  0.99033  0.99630  0.99133   0.9881  0.99117  0.98522   0.9850
# Prevalence            0.09667   0.1067  0.10467  0.09933  0.09067  0.07600   0.1067  0.09867  0.10200   0.1187
# Detection Rate        0.09600   0.1047  0.09533  0.09067  0.08733  0.06800   0.0960  0.09067  0.08867   0.1053
# Detection Prevalence  0.10333   0.1120  0.09933  0.10400  0.10000  0.07733   0.1000  0.09400  0.09800   0.1120
# Balanced Accuracy     0.99249   0.9865  0.95318  0.94897  0.97465  0.94232   0.9478  0.95761  0.92944   0.9400

#--------------------------------------------------------------------
# 4.2 Using RBF Kernel
#--------------------------------------------------------------------

Model_RBF <- ksvm(DIGIT~ ., data = train, scaled = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF,test)
#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test$DIGIT)

# Using Hyperparameter Tuning and Cross Validation
trainControl <- trainControl(method="cv", number=3)
metric <- "Accuracy"
set.seed(7)
grid <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5,1) )
fit.svm <- train(DIGIT~., data=train, method="svmRadial", metric=metric,tuneGrid=grid, trControl=trainControl)
print(fit.svm)
plot(fit.svm)
stopCluster(cl)

#Support Vector Machines with Radial Basis Function Kernel 

#9000 samples
#784 predictor
#10 classes: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 

#No pre-processing
#Resampling: Cross-Validated (3 fold) 
#Summary of sample sizes: 6001, 6000, 5999 
#Resampling results across tuning parameters:

#  sigma  C    Accuracy   Kappa    
#0.025  0.1  0.9233323  0.9148004
#0.025  0.5  0.9572207  0.9524607
#0.025  1.0  0.9632212  0.9591285
#0.050  0.1  0.8374437  0.8193212
#0.050  0.5  0.9483330  0.9425822
#0.050  1.0  0.9599996  0.9555480

#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 0.025 and C = 1.

# Prediction for Testset
Eval_SVMRadial<- predict(fit.svm,test)
confusionMatrix(Eval_SVMRadial,test$DIGIT)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction   0   1   2   3   4   5   6   7   8   9
# 0 145   0   1   0   0   0   2   0   0   3
# 1   0 158   0   0   0   0   1   2   0   3
# 2   0   0 153   0   1   0   0   2   1   0
# 3   0   0   0 147   0   2   0   0   4   1
# 4   0   0   0   0 132   0   1   2   1   4
# 5   0   0   0   0   0 111   1   0   1   1
# 6   0   1   0   0   0   1 153   0   0   0
# 7   0   0   0   2   0   0   0 137   0   1
# 8   0   1   3   0   1   0   2   0 146   0
# 9   0   0   0   0   2   0   0   5   0 165

# Overall Statistics
# 
# Accuracy : 0.9647         
# 95% CI : (0.954, 0.9734)
# No Information Rate : 0.1187         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.9607         
# Mcnemar's Test P-Value : NA             
# 
# Statistics by Class:
# 
# Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity           1.00000   0.9875   0.9745  0.98658  0.97059   0.9737   0.9563  0.92568  0.95425   0.9270
# Specificity           0.99557   0.9955   0.9970  0.99482  0.99413   0.9978   0.9985  0.99778  0.99480   0.9947
# Pos Pred Value        0.96026   0.9634   0.9745  0.95455  0.94286   0.9737   0.9871  0.97857  0.95425   0.9593
# Neg Pred Value        1.00000   0.9985   0.9970  0.99851  0.99706   0.9978   0.9948  0.99191  0.99480   0.9902
# Prevalence            0.09667   0.1067   0.1047  0.09933  0.09067   0.0760   0.1067  0.09867  0.10200   0.1187
# Detection Rate        0.09667   0.1053   0.1020  0.09800  0.08800   0.0740   0.1020  0.09133  0.09733   0.1100
# Detection Prevalence  0.10067   0.1093   0.1047  0.10267  0.09333   0.0760   0.1033  0.09333  0.10200   0.1147
# Balanced Accuracy     0.99779   0.9915   0.9858  0.99070  0.98236   0.9858   0.9774  0.96173  0.97453   0.9608