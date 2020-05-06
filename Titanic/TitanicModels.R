##
## Titanic Analysis using MLR
##

## Libraries
library(tidyverse)

## Read in the Data
titanic.train <- read.csv(file = "../Data/train.csv", stringsAsFactors = FALSE)
titanic.test <- read.csv(file="../Data/test.csv", stringsAsFactors = FALSE)
titanic <- bind_rows(titanic.train, titanic.test)

## Take a quick look at the data
summary(titanic)

###############################
## Data Cleaning / Wrangling ##
###############################

## Remove some variables
titanic <- titanic %>% select(-PassengerId, -Name, -Ticket, -Cabin)

## If Missing Embarkment make an S
titanic$Embarked[titanic$Embarked==""] <- "S"
titanic$Embarked <- as.factor(titanic$Embarked)

## Change Pclass and Sex to a factor
titanic <- within(titanic, {
  Pclass <- as.factor(Pclass)
  Sex <- as.factor(Sex)
})

## Fill in missing ages with MLR prediction
age.lm <- lm(log(Age)~Pclass+Sex+SibSp+Parch+Fare+Embarked, 
             data=titanic)
age.pred <- predict.lm(age.lm, newdata=titanic %>% filter(is.na(Age))) %>% 
  exp() %>% round()
titanic[is.na(titanic$Age),'Age'] <- age.pred

## Fill in missing fare in test set with MLR prediction
fare.lm <- lm(Fare~Pclass+Sex+SibSp+Parch+Age+Embarked,
              data=titanic)
titanic$Fare[is.na(titanic$Fare)] <- predict.lm(fare.lm, 
                                                newdata=titanic[is.na(titanic$Fare),]) %>%
  max(., 0)

########################
## KNN Implementation ##
########################

## Library
library(class)

## Create data matrices
X <- fastDummies::dummy_cols(titanic %>% select(-Survived)) %>% select(-Sex, -Embarked, -Pclass)
y <- as.factor(titanic$Survived)

## Scale the X's so they are all on the same scale
X <- scale(X)

## Standardize Everything (to [0,1]) - a popular standard in many machine learning tutorials
# X <- scale(X, center=apply(X, 2 min), 
#                        scale=apply(X, 2, max)-apply(X, 2, min))

## Split into test and training sets
X.train <- X[!is.na(titanic$Survived),]
X.test <- X[is.na(titanic$Survived),]
y.train <- y[!is.na(y)]

## How good is KNN?
set.seed(17)
test.obs <- sample(1:nrow(X.train), round(0.1*nrow(X.train)))
test.X <- X.train[test.obs,]
train.X <- X.train[-test.obs,]
train.y <- y.train[-test.obs]
test.y <- y.train[test.obs]
acc <- rep(NA, 100)
for(k in 1:length(acc)){
  knn.cv <- knn(train=train.X, cl=train.y, test=test.X, k=k)
  acc[k] <- mean(test.y==knn.cv)
}
qplot(x=1:length(acc), y=acc, geom="line")
best.k <- which.max(acc)
best.k

## Prediction with KNN
knn.preds <- knn(train=X.train, cl=y.train, test=X.test, k=best.k, prob=TRUE)
knn.preds
attr(knn.preds, "prob")

#############################
## Naive Bayes Classifiers ##
#############################

## Library
library(naivebayes)

## Fit a naive Bayes model
nb.model <- naive_bayes(as.factor(Survived)~., 
                        data=titanic %>% filter(!is.na(Survived)), 
                        usekernel=TRUE)

## Generate predictions
nb.preds <- predict(nb.model, 
        newdata=titanic %>% filter(is.na(Survived)) %>% select(-Survived))


##########################
## Classification Trees ##
##########################

## Libraries
library(rpart)
library(rpart.plot)

## Fit a tree
titanic.tree <- rpart(as.factor(Survived)~., 
      data=titanic %>% filter(!is.na(Survived)),
      control=list(cp=0), method="class")
rpart.plot(titanic.tree)
plotcp(titanic.tree)

## Prune it back
titanic.tree.pruned <- prune(titanic.tree, cp=titanic.tree$cptable[,'CP'][which.min(titanic.tree$cptable[,'xerror'])])
rpart.plot(titanic.tree.pruned, box.palette=c("red","green"))


####################
## Random Forests ##
####################

## Library
library(randomForest)

## Fit a Random Forest
titanic.train.sub <- sample(nrow(titanic.train), round(0.9*nrow(titanic.train)))
titanic.trainset <- titanic %>% filter(!is.na(Survived))
titanic.train.use <- titanic.trainset[titanic.train.sub,]
titanic.train.test <- titanic.trainset[-titanic.train.sub,]
titanic.rf <- randomForest(as.factor(Survived)~.,
                           data=titanic.trainset,
                           mtry=3,
                           ntree=500,
                           importance=TRUE)
titanic.rf <- randomForest(x=titanic.train.use %>% select(-Survived),
                           y=as.factor(titanic.train.use[['Survived']]),
                           xtest=titanic.train.test %>% select(-Survived),
                           ytest=as.factor(titanic.train.test$Survived),
                           mtry=3,
                           ntree=500,
                           importance=TRUE)
plot(titanic.rf)                        
varImpPlot(titanic.rf)
predict(titanic.rf, newdata=titanic %>% filter(is.na(Survived)))

#############################
## Support Vector Machines ##
#############################

## Library
library(e1071) # SVMs

## I am finding that feeding the function the X-matrix
## without an intercept is substantially more stable.  Using
## formulas results in weird factor errors
svm.X <- fastDummies::dummy_cols(titanic %>% select(-Survived)) %>% select(-Sex, -Embarked, -Pclass)
svm.y <- as.factor(titanic$Survived)

## Standardize Everything (to [0,1]) - a popular standard in many machine learning tutorials
svm.X <- scale(svm.X, center=apply(svm.X, 2, min),
                       scale=apply(svm.X, 2, max)-apply(svm.X, 2, min))

## Split into test and training sets
X.train.svm <- svm.X[!is.na(titanic$Survived),]
X.test.svm <- svm.X[is.na(titanic$Survived),]
y.train.svm <- svm.y[!is.na(svm.y)]

## Split training into validation sets
valid <- sample(1:nrow(X.train.svm), round(0.1*nrow(X.train.svm))) %>% sort()

## Tune a support vector machine to gamma (kernel parameter)
## and cost (penalty)
titanic.svm.tune <- tune(svm, train.x=X.train.svm[-valid,], train.y=y.train.svm[-valid],
                        validation.x=X.train.svm[valid,], validation.y=y.train.svm[valid],
                        ranges=list(cost=seq(0.05,10, length=25), gamma=seq(0.1, 1,length=25)),
                        kernel="radial") ## Takes a few min to run

## Plot Performance as a function of gamma and cost
ggplot(data=titanic.svm.tune$performances, aes(x=gamma, y=cost, fill=error)) + 
  geom_raster() + scale_fill_gradient(low="blue", high="red")

## What are the best settings to use
titanic.svm.tune$performances[order(titanic.svm.tune$performances[['error']])[1:10],]

## Fit the "best" svm
titanic.svm <- svm(x=X.train.svm, y=y.train.svm, kernel="radial", 
                  cost=titanic.svm.tune$best.parameters['cost'], 
                  gamma=titanic.svm.tune$best.parameters['gamma'],
                  probability=TRUE)

## Predict using the SVM
svm.preds <- predict(titanic.svm, newdata=X.test.svm, probability=TRUE)


############################
## Fitting Neural Network ##
############################

## Install Keras, specify python path and load library
## devtools::install_github("rstudio/keras")
library(keras) #Neural Networks
use_python("/Users/mheaton/opt/anaconda3/bin/python")
## keras::install_keras(method="conda")

## Break into X matrix of dummy variables
X.nn <- fastDummies::dummy_cols(titanic %>% select(-Survived)) %>% select(-Pclass, -Sex, -Embarked)

## Normalize the data to [0,1]
X.nn <- scale(X.nn, center=apply(X.nn, 2, min), scale=apply(X.nn, 2, max)-apply(X.nn, 2, min))
X.nn.train <- X.nn[!is.na(titanic$Survived),]
X.nn.test <- X.nn[is.na(titanic$Survived),]

## Break Y into matrix of 0's and 1's
Y <- fastDummies::dummy_cols(titanic['Survived']) %>% select(-Survived)
Y.train <- Y[Y$Survived_NA==0,1:2] %>% as.matrix()

## Further split train to check accuracy
train.obs <- sample(nrow(Y.train), round(0.9*nrow(Y.train)))

## Define a NN
nn.model <- keras_model_sequential() #Initialize
nn.model %>%
  layer_dense(units=4, input_shape=ncol(X.nn.train)) %>%
  layer_activation("relu") %>%
  layer_dense(units=2) %>%
  layer_activation("softmax")

## Define how to fit a NN
nn.model %>% compile(optimizer=optimizer_sgd(lr=0.01),
                                 loss="binary_crossentropy",
                                 metrics="accuracy")

## Now fit a NN
fit.history <- nn.model %>% fit(x=X.nn.train[train.obs,],
                             y=Y.train[train.obs,],
                             batch_size=100,
                             epochs=100,
                             validation_split=0.1,
                             verbose=0)
plot(fit.history)

## Generate Predictions
preds <- nn.model %>% predict_proba(x=X.nn.train[-train.obs,])
pred.class <- nn.model %>% predict_classes(x=X.nn.train[-train.obs,])




