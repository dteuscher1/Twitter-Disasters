# Twitter Disaster NLP Project

This repository includes exploratory analysis and predictive modeling for the Real or Not? NLP with Disaster Tweets Kaggle competition (https://www.kaggle.com/c/nlp-getting-started).

The .Rmd file were the analysis was done is available, while an HTML document `nlp_approaches.html` has been created that includes the code and output that was done. A notebook was created on Kaggle for this competition as well and can be found [here](https://www.kaggle.com/davidkt2/twitter-nlp-eda-and-model-fit).

The goal of this competition is to determine whether or not a tweet is about a real disaster. There is a lot of flexibility to the human language that machines can't always determine, so the goal is to build a machine learning model that can determine some of the nuances about whether or not a tweet is about a real disaster. The dataset provided includes id for the tweet, the text of the tweet, any keywords or location associated with the tweet. The training set also includes an additional column, `target`, that indicates whether or not the tweet is about a real disaster. There are 7613 observations in the training set, while there are 3263 observations in the test set. 

The best model that was fit to the data was a logistic regression model with an L1 penalty and the mean F-1 score was .79650, which is approximately the 51st percentile in the competition. Since it is the first experience using NLP methods and algorithms, it isn't a bad start, but there is definitely room for improvement. Some potential improvements include attempting to better tune the hyperparameters for the different models, as well as trying models that often do very well like `gbm` and `xgboost`. Additional research into NLP methods and techniques would also likely increase the predictive accuracy of models that would be fit. 

If there are any questions or comments about the analysis and work done, feel free to email me at david.teuscher.96@gmail.com