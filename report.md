---
title: "Practical Machine Learning Project"
author: "Reza Rosli (zaros)"
date: "26 October 2015"
output: html_document
---

# Executive Summary 

In this project, we obtained data from a set of accelerometers on the belt, forearm, arm, and dumbell of 6 participants.The participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fash- ions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. 

Here we report on how we used the dataset to create a random forest model to predict the classification of test cases in a test data set, i.e. predict the manner in which they performed a particular exercise. It was found that the generated model was able to predict the classification to 97.92% accuracy and an out of bag estimate of error rate of 1.79%.

# Modeling

## Loading and Cleaning Data

The data used for the analysis is included in the Git repo was originally downloaded from the following locations:

- Training Data, https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
- Test Data, https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
# libraries
library(caret)
set.seed(286)

# Loading data
data <- read.csv('pml-training.csv', na.strings=c("NA","","#DIV/0!"), strip.white=T)
data.test <- read.csv('pml-testing.csv', na.strings=c("NA","","#DIV/0!"), strip.white=T)
```

It was necessary to perform some cleaning to the data: 

- There were a number of columns which consists of mostly N/A values, which were removed as they cannot possibly contribute any valuable information to our model.  
- A set of columns which are identifiers and not measurements such as the subject's name and test timestamp were removed.


```r
# remove columns which are mostly NA
isNA <- apply(data, 2, function(x) { sum(is.na(x)) })
data <- data[,isNA==0]

# remove columns which are not relevant
data <- data[,-(1:7)]

# repeat for the test (validation) dataset
isNA <- apply(data.test, 2, function(x) { sum(is.na(x)) })
data.test <- data.test[,isNA==0]
data.test <- data.test[,-(1:7)]
```

## Training the Model

It was found that the random tree training algorithm was liable to take an unacceptable amount of time for the purpose of this project. To manage the computational time needed, only 50% of the provided dataset is used to create the model. 


```r
# select only a proportion of the dataset for training/testing
subset.pct = 0.5
data <- data[sample(nrow(data), subset.pct*nrow(data)), ]
```

Since we will be using Random Forest, we do not need actually need to partition a separate testing dataset, since the random forest algorithm will automatically split the datasets in each of its iterations. However, we make a 70/30 split here so that we can make a test prediction on a separate step later.


```r
# Split the training data to training and test sets
inTrain <- createDataPartition(y=data$classe,p=0.70,list=FALSE)
training <- data[inTrain,]
testing<-data[-inTrain,]
```

We will use the Random Forest method because of it's expected high accuracy. To avoid overfitting, we use `k`-fold cross-validation with `k`=5.


```r
# train using random forest
# proximity = FALSE to cut down on computation time
ctrl <- trainControl(method="cv", number=5)
model <- train(classe ~ ., data=training, model="rf", trControl=ctrl,proximity=FALSE,allowParallel=TRUE)
```

## The Resulting Model & Test Prediction

The result of the training created a Random Forest model with `ntree`=500 and `mtry`=27. The model is expected to be 97.92% accurate with 1.79% OOB error rate. 


```r
model
```

```
## Random Forest 
## 
## 6870 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 5495, 5498, 5496, 5495, 5496 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9775825  0.9716533  0.001901726  0.002400192
##   27    0.9791836  0.9736746  0.002616077  0.003305427
##   52    0.9735060  0.9664999  0.005662486  0.007152199
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = FALSE,      model = "rf", allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 1.79%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 1933    6    2    1    0 0.004634398
## B   28 1272   12    3    0 0.032699620
## C    1   23 1196    8    0 0.026058632
## D    0    2   17 1116    1 0.017605634
## E    0    3    7    9 1230 0.015212170
```

Validating the model against the test partition we created earlier found that the model is observed to be 98.6% accurate.


```r
# testing the model against testing data
pred <- predict(model, newdata=testing)
cm <- confusionMatrix(testing$classe,pred)
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 831   1   0   0   0
##          B   5 553   5   0   0
##          C   0  16 509   0   0
##          D   0   1   9 476   0
##          E   0   0   1   3 531
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9861        
##                  95% CI : (0.9811, 0.99)
##     No Information Rate : 0.2843        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9824        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9940   0.9685   0.9714   0.9937   1.0000
## Specificity            0.9995   0.9958   0.9934   0.9959   0.9983
## Pos Pred Value         0.9988   0.9822   0.9695   0.9794   0.9925
## Neg Pred Value         0.9976   0.9924   0.9938   0.9988   1.0000
## Prevalence             0.2843   0.1942   0.1782   0.1629   0.1806
## Detection Rate         0.2826   0.1880   0.1731   0.1618   0.1806
## Detection Prevalence   0.2829   0.1914   0.1785   0.1652   0.1819
## Balanced Accuracy      0.9968   0.9821   0.9824   0.9948   0.9992
```

# Conclusions

The model was used to predict the classifiers for the test dataset, as below:


```r
answers <- predict(model,newdata=data.test)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


# References
1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. Read more: http://groupware.les.inf.puc-rio.br/har
