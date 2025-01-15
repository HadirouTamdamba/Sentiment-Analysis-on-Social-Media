# Sentiment Analysis on Social Media 

## Problem Statement
The goal is to analyze user sentiment from tweets about airlines. This project is useful for companies looking to understand public opinion.

## Dataset Description 
The **Twitter US Airline Sentiment** dataset contains 14,640 tweets with the following variables:
- **text**: Tweet content.
- **airline_sentiment**: Sentiment (positive, negative, neutral).

## Exploratory Data Analysis (EDA)
- Distribution of sentiments.
- Word cloud for negative tweets.

## Modeling
Comparison of several models:
- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Random Forest**


## Evaluation
The best model is selected based on **accuracy**.

## Deployment
A Flask API is created to analyze sentiment in real-time.

## Conclusion
The **Logistic Regression** model achieved the best performance with an accuracy of 0.80.This project demonstrates how AI can help understand public opinion.

