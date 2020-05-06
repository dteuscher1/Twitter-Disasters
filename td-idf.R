library(tidytext)
library(tidyverse)
library(text2vec)

twitter <- read_csv('train.csv')

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
vocab <- create_vocabulary(it_train)

vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
tfidf <- TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)

t1 <- Sys.time()
library(glmnet)
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf, y = twitter$target, 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 4,
                              thresh = 1e-3,
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))

plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
