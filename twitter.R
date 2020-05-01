###########
# Twitter #
###########

library(tidyverse)

twitter <- read_csv("train.csv")

### Libraries ###
library(stringr)

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
head(twitter)

# Numbers
twitter$numbers <- str_count(twitter$text, "[0-9]")

# Naive Bayes: David 
# SVM: Matt
# Random Forest: Shane
# KNN: McKay

# Replacing NAs with None...
twitter$keyword <- replace_na(twitter$keyword, replace = "none")
twitter$location <- replace_na(twitter$keyword, replace = "None")

# Making the target variable a factor
twitter$target <- as.factor(twitter$target)

twitter$target <- revalue(twitter$target, c("0" = "N", "1" = "Y"))

# Re-ordering columns. Unsure if necessary
twitter <- twitter[, c(1:4, 6:12, 5)]

# Naive Bayes: David 
# SVM: Matt
# Random Forest: Shane
# KNN: McKay

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using the following function
                           summaryFunction = twoClassSummary)

# Support Vector Machine
svmFit <- train(target ~ . -id -text, 
                data = twitter, 
                method = "svmRadial", 
                trControl = fitControl, 
                preProc = c("center", "scale"),
                tuneLength = 4,
                metric = "ROC")
svmFit 