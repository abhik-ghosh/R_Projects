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

# Load required Librbaries: 
library(kernlab)
library(readr)
library(caret)
library(caTools)
library(e1071)

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

# Scaling data 

max(train[ ,2:ncol(train)]) # max pixel value is 255, lets use this to scale data
train[ , 2:ncol(train)] <- train[ , 2:ncol(train)]/255
test <- cbind(label = test_data[ ,1], test_data[ , 2:ncol(test_data)]/255)
x<-test[,-1]
#####################################################################################

# 4. Model Building

#--------------------------------------------------------------------
# 4.1 Using Linear Kernel
#####################################################################

Model_linear <- ksvm(DIGIT~ ., data = train, scaled = FALSE, kernel = "vanilladot", c=1)
Eval_linear<- predict(Model_linear, test)

model1_linear <- ksvm(label ~ ., data = train, scaled = FALSE, kernel = "vanilladot", C = 1)
print(model1_linear) 

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test_data$DIGIT)

#--------------------------------------------------------------------
# 4.2 Using RBF Kernel
#####################################################################

#Using RBF Kernel
Model_RBF <- ksvm(letter~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test_data)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test_data$DIGIT)

