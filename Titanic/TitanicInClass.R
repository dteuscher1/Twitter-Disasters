##
## Approaches for the titanic data
##

###################
## Preliminaries ##
###################

## Libraries that I need
library(tidyverse)

## Read in the data and merge into single data frame
titanic.train <- read.csv(file = "train.csv", stringsAsFactors = FALSE)
titanic.test <- read.csv(file="test.csv", stringsAsFactors = FALSE)
titanic <- bind_rows(titanic.train, titanic.test)

## Take a quick look at the data
summary(titanic)

###############################
## Data Cleaning / Wrangling ##
###############################

## Remove some variables
titanic <- titanic %>% dplyr::select(-PassengerId, -Name, -Ticket, -Cabin)

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


#### KNN Implementation

library(class)

## Create data matricies
X <- fastDummies::dummy_cols(titanic %>% dplyr::select(-Survived)) %>% dplyr::select(-Sex, -Embarked, -Pclass)
y <- as.factor(titanic$Survived)

# Scale the X's so they are all on the same scale
X <- scale(X)

## Split into test and training

X.train <- X[!is.na(titanic$Survived),]
X.test <- X[is.na(titanic$Survived), ]
y.train <- y[!is.na(y)]


# How good is this 
set.seed(17)
test.obs <- sample(1:nrow(X.train), round(.1*nrow(X.train)))
test.X <- X.train[test.obs, ]
train.X <- X.train[-test.obs,]
train.y <- y.train[-test.obs]    
test.y <- y.train[test.obs]
acc <- rep(NA, 100)
for(k in 1:length(acc)){
knn.cv <- knn(train = train.X, cl = train.y, test = test.X, k = k)
acc[k] <- mean(test.y == knn.cv)
}
qplot(x = 1:length(acc), y = acc, geom='line')
knn.preds <- knn(train = X.train, cl = y.train, test = X.test, k = 10, prob = TRUE)
knn.preds
