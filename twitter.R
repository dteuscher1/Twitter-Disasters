###########
# Twitter #
###########

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

# filling missing values
twitter$keyword[is.na(twitter$keyword)] <- "None"
twitter$location[is.na(twitter$location)] <- "None"

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

# Naive Bayes

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
  summaryFunction = twoClassSummary,
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
