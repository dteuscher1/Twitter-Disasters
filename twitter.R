###########
# Twitter #
###########

library(tidyverse)

twitter <- read_csv("train.csv")

### Libraries ###
library(stringr)
library(sentimentr)

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