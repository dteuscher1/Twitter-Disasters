library(tidyverse)
library(text2vec)
library(glmnet)
library(corpus)
library(stringr)
library(sentimentr)


twitter <- read_csv('train.csv')
twitter_test <- read_csv('test.csv')

set.seed(2020)

prep_fun <- tolower
tok_fun <- word_tokenizer

# twit.1 <- rbind(twitter, cbind(twitter_test, "target" = rep(NA, nrow(twitter_test)))) # <- use if using test data set to create vector space
# twitter


#Logistic Regression
twit_train <- itoken(twitter$text, #<- change to twit.1 if using test data set to create vector space
                     preprocessor = tolower,
                     tokenizer = word_tokenizer,
                     ids = twitter$id,
                     progressbar = TRUE)
vocab <- create_vocabulary(twit_train)
vectorizer <- vocab_vectorizer(vocab) #creates vector space
t1 <- Sys.time()
# twit_train.1 <- itoken(twitter$text, #<- change to twit.1 if using test data set to create vector space
#                      preprocessor = tolower,
#                      tokenizer = word_tokenizer,
#                      ids = twitter$id,
#                      progressbar = TRUE)

# as is - .8488
# no urls - .8485
# with additional variables - .8533
# with test data in vector space - .8532
# with test data in vector space and additional variables - 0.8532

dtm_train <- create_dtm(twit_train, vectorizer) #creating document text matrix
print(difftime(Sys.time(), t1, units = "sec"))

t1 <- Sys.time()
glmnet.classifier.log <- cv.glmnet(x = cbind(dtm_train, as.matrix(additional.matrix)),
                                   y = as.factor(twitter$target),
                                   family = "binomial",
                                   alpha = 1, 
                                   type.measure = "auc",
                                   nfolds = 4, 
                                   thresh = 1e-3,
                                   maxit = 1e3)
print(difftime(Sys.time(), t1, units = "sec"))
plot(glmnet.classifier.log)
print(paste("max AUC =", round(max(glmnet.classifier.log$cvm), 4)))

twit_test <- twitter_test$text %>% prep_fun %>% tok_fun %>% itoken(., ids = twitter_test$id)
dtm_test <- create_dtm(twit_test, vectorizer)

# preds.log.2 <- predict(glmnet.classifier.log, dtm_test, type = "class")[,1]
preds.log <- predict(glmnet.classifier.log, dtm_test, type = "response")[,1]

preds.log.out2 <- cbind("id" = twitter_test$id, as.integer(preds.log > 0.45))
write_csv(as.data.frame(preds.log.out2), 'preds.log.out2.csv')


#cross validation
set.seed(2020)
test.set <- sample( 1:nrow(twitter), 0.1*nrow(twitter))
twit.train <- twitter[-test.set,]
twit.test <- twitter[test.set,]

cv_train <- itoken(twit.train$text,
                     preprocessor = tolower,
                     tokenizer = word_tokenizer,
                     ids = twit.train$id,
                     progressbar = TRUE)
vocab <- create_vocabulary(cv_train)
vectorizer <- vocab_vectorizer(vocab) 
dtm_train <- create_dtm(cv_train, vectorizer) #creating document text matrix
glmnet.classifier.cv <- cv.glmnet(x = dtm_train,
                                   y = as.factor(twit.train$target),
                                   family = "binomial",
                                   alpha = 1, 
                                   type.measure = "auc",
                                   nfolds = 4, 
                                   thresh = 1e-3,
                                   maxit = 1e3)
cv_test <- twit.test$text %>% prep_fun %>% tok_fun %>% itoken(., ids = twit.test$id)
dtm_test <- create_dtm(cv_test, vectorizer)

cv.perc <- c()
for (i in seq(.3, .7, by = 0.01)) {
  print(i)
  cv.perc <- append(cv.perc, mean((as.integer(as.numeric(predict(glmnet.classifier.cv, dtm_test, type = "response")[,1]) > i))  == twit.test$target))
  
}
seq(.3, .7, by = 0.01)[which.max(cv.perc)]


mean(preds.log.out[,2] == submission_word2vec[,2])
write_csv(as.data.frame(preds.log.out), 'preds.log.out.csv')



preds.log.out <- cbind("id" = twitter_test$id, as.integer(preds.log > 0.45))


## Adjustments
twit_train <- itoken(twitter$text, #<- change to twit.1 if using test data set to create vector space
                     preprocessor = tolower,
                     tokenizer = word_tokenizer,
                     ids = twitter$id,
                     progressbar = TRUE)
vocab <- create_vocabulary(twit_train, stopwords = stopwords_en)
pruned_vocab <- prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
vectorizer <- vocab_vectorizer(pruned_vocab) 

dtm_train  <- create_dtm(twit_train, vectorizer)
dim(dtm_train)
glmnet_classifier <- cv.glmnet(x = dtm_train, y = twitter$target, 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 10,
                              thresh = 1e-3,
                              maxit = 1e3)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

# Adding additional variables
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
head(twitter)
add.variables <- twitter %>% select(6:15) %>% as.matrix()
mat.variables <- cbind(dtm_train, add.variables)

glmnet_classifier1 <- cv.glmnet(x = mat.variables, y = twitter$target, 
                               family = 'binomial', 
                               alpha = 1,
                               type.measure = "auc",
                               nfolds = 10,
                               thresh = 1e-3,
                               maxit = 1e3)
plot(glmnet_classifier1)
print(paste("max AUC =", round(max(glmnet_classifier1$cvm), 4)))
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
#ensenble, try adding our additional variables, removing stop-words, playing with model parameters