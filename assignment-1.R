# libraries
library(caret)
set.seed(286)

# Set the working directory
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

# Loading data
data <- read.csv('pml-training.csv', na.strings=c("NA","","#DIV/0!"), strip.white=T)
data.test <- read.csv('pml-testing.csv', na.strings=c("NA","","#DIV/0!"), strip.white=T)

# Data Prep
# remove columns which are mostly NA
isNA <- apply(data, 2, function(x) { sum(is.na(x)) })
data <- data[,isNA==0]

# remove columns which are not relevant
data <- data[,-(1:7)]

# repeat for the test (validation) dataset
isNA <- apply(data.test, 2, function(x) { sum(is.na(x)) })
data.test <- data.test[,isNA==0]
data.test <- data.test[,-(1:7)]

# because we're short on time, select only a proportion
# of the dataset for training/testing
subset.pct = 0.5
data <- data[sample(nrow(data), subset.pct*nrow(data)), ]

# Split the training data to training and test sets
inTrain <- createDataPartition(y=data$classe,p=0.70,list=FALSE)
training <- data[inTrain,]
testing<-data[-inTrain,]

# train using random forest
# proximity = FALSE to cut down on computation time
ctrl <- trainControl(method="cv", number=5)
model <- train(classe ~ ., data=training, model="rf", trControl=ctrl,proximity=FALSE,allowParallel=TRUE)

# testing the model against testing data
pred <- predict(model, newdata=testing)
cm <- confusionMatrix(testing$classe,pred)


# Write out answers for submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./answers/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

answers <- predict(model,newdata=data.test)
answers
pml_write_files(answers)
