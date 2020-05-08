library(tidytext)
library(tidyverse)
library(text2vec)

twitter <- read_csv('train.csv')
twitter_test <- read_csv('test.csv')


tweets <- twitter %>% 
  unnest_tokens(word, text) %>% 
  anti_join(stop_words)

tweets_tf_idf <- tweets %>% 
  count(target, word, sort = TRUE) %>%
  ungroup() %>%
  bind_tf_idf(word, target, n)

tweets_tf_idf %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>% 
  group_by(target) %>% 
  top_n(15) %>% 
  ungroup() %>%
  ggplot(aes(word, tf_idf, fill = target)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~target, ncol = 2, scales = "free") +
  coord_flip()


prep_fun <- tolower
tok_fun <-  word_tokenizer

it_train <- itoken(twitter$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = twitter$id, 
                  progressbar = TRUE)
it_test <- itoken(twitter_test$text,
                  preprocessor = prep_fun,
                  tokenizer = tok_fun,
                  ids = twitter_test$id,
                  progressbar = TRUE)
vocab <- create_vocabulary(it_train, c(1L, 1L))

vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
tfidf <- TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)

dtm_test_tfidf = create_dtm(it_test, vectorizer)
dtm_test_tfidf = transform(dtm_test_tfidf, tfidf)
t1 <- Sys.time()
library(glmnet)
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf, y = twitter$target, 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 8,
                              thresh = 0.0001,
                              maxit = 10000)
print(difftime(Sys.time(), t1, units = 'sec'))

plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

preds <- predict(glmnet_classifier, dtm_test_tfidf, type = "class")
submission <- tibble(id=twitter_test$id, target=as.numeric(preds))



## Skyler's attempt
lambda_hat <- glmnet_classifier$lambda.1se

twitter_test <- read_csv('test.csv')
it_test <- itoken(twitter_test$text, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun, 
                   ids = twitter_test$id, 
                   progressbar = TRUE)
vocab <- create_vocabulary(it_test, c(1L, 2L))

dtm_test <- create_dtm(it_test, vectorizer)
dtm_test_tfidf <- transform(dtm_test, tfidf)

model <- glmnet(x = dtm_train_tfidf, y = twitter$target, 
                family = 'binomial', lambda=lambda_hat)
y_hat <- predict(model, dtm_test_tfidf)
y_hat <- as.integer(y_hat > 0)
submission_word2vec <- tibble(id=twitter_test$id, target=y_hat)
write_csv(submission_word2vec, 'submission_word2vec.csv')
