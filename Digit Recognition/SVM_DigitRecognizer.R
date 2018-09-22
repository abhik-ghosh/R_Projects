setwd("C:/Users/abhik/Desktop/Work Folder/R Projects/Digit Recognition")

############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

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
library(doParallel)

train_data <- read.csv("mnist_train.csv", stringsAsFactors = F)
test_data <- read.csv("mnist_test.csv", stringsAsFactors = F)
dim(train_data)
dim(test_data)
# Train Data : 
# Number of Instances: 59,999
# Number of Attributes: 785 
# Test Data : 
# Number of Instances: 9,999
# Number of Attributes: 785

str(train_data)
summary(train_data)
#View(train_data)
head(train_data)


#####################################################################################

#3. Data Preparation:

# The first column seems to be the label of the datasets
names(train_data)[1] <- "DIGIT"
names(test_data)[1] <- "DIGIT"
colnames(train_data)
colnames(test_data)

# take subset of common columns
common_cols <- intersect(colnames(train_data), colnames(test_data))
train_data <- select(train_data, common_cols)
test_data <- select(test_data, common_cols)

# Changing output variable "DIGIT" to factor type 
train_data$DIGIT <- as.factor(train_data$DIGIT)
test_data$DIGIT <- as.factor(test_data$DIGIT)

# Checking missing value
sapply(train_data, function(x) sum(is.na(x))) # No missing values
sapply(test_data, function(x) sum(is.na(x))) # No missing values

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

#--------------------------------------------------------------------
# 4.1 Using Linear Kernel
#####################################################################

Model_linear <- ksvm(DIGIT~ ., data = train, scaled = FALSE, kernel = "vanilladot")
print(Model_linear) 
Eval_linear<- predict(Model_linear, test)


#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$DIGIT)

#--------------------------------------------------------------------
# 4.2 Using RBF Kernel
#####################################################################

#Using RBF Kernel
Model_RBF <- ksvm(DIGIT~ ., data = train, scaled = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF,test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test$DIGIT)

############   Hyperparameter tuning and Cross Validation #####################


trainControl <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5,1,2) )

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(DIGIT~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

stopCluster(cl)

print(fit.svm)

plot(fit.svm)