library(tidytext)
library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(scales)

twitter <- read_csv('train.csv')
tidy_twit <- twitter %>%
  mutate(text = gsub("http(.*)", "http", text)) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  mutate(word = str_extract(word, '[\\da-z\']+'))

frequency <- tidy_twit %>%
  group_by(target) %>%
  count(word) %>%
  mutate(proportion = n/sum(n)) %>%
  select(-n) %>%
  spread(target, proportion) 

# Expect warning messages about missing values
ggplot(frequency, aes(x = `0`, y = `1`, color = abs(`1` - `0`))) +
  geom_abline(color = "gray40", lty = 2) +
  geom_jitter(alpha = 0.1, size = 2.5, width = 0.3, height = 0.3) +
  geom_text(aes(label = word), check_overlap = TRUE, vjust = 1.5) +
  scale_x_log10(labels = percent_format()) +
  scale_y_log10(labels = percent_format()) +
  scale_color_gradient(limits = c(0, 0.001), low = "darkslategray4", high = "gray75") +
  theme(legend.position="none") +
  labs(title="Word Frequency (in Proportions)\nBetween Tweet Message Types", y="Disaster Tweet", x="Irrelevant Tweet")

