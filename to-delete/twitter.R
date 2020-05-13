###########
# Twitter #
###########

# Naive Bayes: David 
# SVM: Matt
# Random Forest: Shane
# KNN: McKay

### Libraries ###
library(stringr)
library(sentimentr)
library(tidyverse)
library(caret)

# Read in training data
twitter <- read_csv("train.csv")


# Specific words within text
# #, urls, including images
# length of texts
# # number of capital letters, exlamation marks, question marks
# Tone of the message - along with key words
# emojis

#1 Punctuation - .!?,"'-
twitter$url_count <- str_count(twitter$text, "http(.*)$")

# Removed urls
twitter$text <- gsub("http(.*)$", "http", twitter$text)

twitter$punct_count <- str_count(twitter$text, "[.!?,\"'-]" )

#2 Handles - @
twitter$handles_count <- str_count(twitter$text, "[@]" )

#3 Hashtag - #
twitter$hashtag_count <- str_count(twitter$text, "[#]" )

#4 Length
twitter$characters <- nchar(twitter$text)

# Capital letter
twitter$capital <- str_count(twitter$text, "[A-Z]")

# Numbers
twitter$numbers <- str_count(twitter$text, "[0-9]")

# Tone
sentiment_df <- sentiment_by(get_sentences(twitter$text))

twitter$tone <- sentiment_df$ave_sentiment

# Word count
twitter$word <- sentiment_df$word_count

# Proportion of capital to lower case letters
twitter <- twitter %>% mutate("cap.prop" = capital/characters)

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using the following function
                           summaryFunction = twoClassSummary)

# SUPPORT VECTOR MACHINE
# filling missing values
twitter$keyword[is.na(twitter$keyword)] <- "None"
twitter$location[is.na(twitter$location)] <- "None"

# Making the target variable a factor
twitter %>% mutate(target = if_else(target=='1', 'Y', 'N'))

# Re-ordering columns. Unsure if necessary
twitter <- twitter[, c(1:4, 6:12, 5)]

svmFit <- train(target ~ . -id -text, 
                data = twitter, 
                method = "svmRadial", 
                trControl = fitControl, 
                preProc = c("center", "scale"),
                tuneLength = 4,
                metric = "ROC")
svmFit 


# NAIVE BAYES
# Add indicator variables for keyword and location
twitter1 <- twitter %>%
  mutate(target = factor(ifelse(target == 1, "Yes", "No"), levels = c("No", "Yes")),
         keyword_ind = ifelse(is.na(keyword), 0, 1),
         location_ind = ifelse(is.na(location), 0, 1))

# Create data frame of predictor variables
x <- twitter1 %>% select(-id, -text, -target, -keyword, -location) %>% as.data.frame()
# Create vector of the response variable
y <- twitter1$target

# Specifies the type of cross validation and to return AUC, sensitivity, and specificity
myControl <- trainControl(
  method="cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Creates a grid to test different values of hyperparameters
grid <- expand.grid(laplace=seq(0,10, length = 5), usekernel=c(TRUE,FALSE), adjust=seq(1,10, length = 5))

# Fit of the Naive Bayes model
nb.model <- train(
  x=x,
  y=y,
  method = "naive_bayes",
  trControl = myControl,
  tuneGrid = grid,
  metric="ROC"
)

nb.model
summary(nb.model)

# Show a plot comparing the models with different hyperparameter values
plot(nb.model)  

##################
# Random Forests #
##################

library(randomForest)

# Indicator for non-NA locations and keywords
twitter$keywordInd <- !is.na(twitter$keyword)
twitter$locationIng <- !is.na(twitter$location)

# I didn't use the first few columns (id, keyword, location, text)
twitter.clean <- twitter[,-c(1:4)]
twitter.clean$target <- as.factor(twitter.clean$target)

# Subsetting to creating training and testing sets
twitter.sub <- sample(nrow(twitter), round(0.9*nrow(twitter)))
twitter.train.use <- twitter.clean[twitter.sub,]
twitter.train.test <- twitter.clean[-twitter.sub,]

# Random Forest Model
twitter.rf <- randomForest(target~.,
                           data=twitter.train.use,
                           mtry=5,
                           ntree=800,
                           importance=TRUE)

# RF plots we did with Heaton, but I forgot what they mean lol
plot(twitter.rf)
varImpPlot(twitter.rf)

# Prediction Assessment (I got around .72)
twitter.train.test$predict <- predict(twitter.rf, newdata=twitter.train.test)
sum(twitter.train.test$target == twitter.train.test$predict) / nrow(twitter.train.test)
