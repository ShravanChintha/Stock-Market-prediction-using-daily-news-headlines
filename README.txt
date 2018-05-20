#README file

#Stock Market prediction using daily news headlines

Date: 4/30/2018
#Authors:
Shravan Chintha - Role: Cleaning dataset, created relevant visualizations, computed baseline model, implemented TFIDF for data. 
Implemented Random forest, naive bayes models.
Adithya Job - Role: Implemented sentiment analysis, bigram and tri gram models with machine learning models such as gradient boosting,
XGboost model and evaluated the model performance.

Problem Definition:

Using historical news, predicting the stock market movement is the problem solved.

1. Introduction:
The project is about predicting the stock market movement based on the news headlines 
that published on a particular day. The news data is collected from Reddit news and top 25
headlines, ranked based on reddit user votes, are taken on each day. The stock market data,
DJIA (Dow Jones Industrial Average) of each day is collected from Yahoo finance. Combined 
both datasets to process and apply modeling techniques further to get desired results.
Different NLP techniques and machine learning models are used to address the problem defined.

2. Outline of solution:
Natural language techniques such as word clouds, bag of words, ngrams, sentiment analysis 
etc., are used to process the data. Also, machine learning techniques such as logistic 
regression, random forest, Naïve Bayes model, gradient boosting, xgboost are applied to 
predict the outcome variable. A baseline model, logistic regression with bag of words is 
performed to check the accuracy and use it as a baseline for our rest analysis. Then, 
computed accuracies of various model to know the best performing model among the models 
we applied. We got the best accuracy for sentiment analysis with xgboost algorithm, able 
to improve the accuracy to 62.7% compared to the baseline 46.07%.


3. Examples of program input and output:
Following the steps mentioned in the project.py python file would serve as input to the 
program. And output of the program is the results of each model such as mentioned below:

Baseline model:46.07%
Logistic Regression with bigrams:53.11%
Random Forest with bigrams:54.52%
Naïve Bayes with bigrams:52.91%
Gradient Boosting machines with bigrams:55.33%
Logistic Regression with trigrams:51.71%
Sentiment analysis:62.77%



