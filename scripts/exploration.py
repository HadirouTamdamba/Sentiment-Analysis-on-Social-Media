import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud  

# Load data
data = pd.read_csv("data/Tweets.csv")

# Distribution of sentiments
plt.figure(figsize=(10, 6)) 
sns.countplot(x="airline_sentiment", data=data)
plt.title("Distribution of sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Number of tweets")
plt.show()

# Word cloud for negative tweets   
negative_tweets = " ".join(data[data["airline_sentiment"] == "negative"]["text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_tweets)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word cloud for negative tweets")
plt.show() 
