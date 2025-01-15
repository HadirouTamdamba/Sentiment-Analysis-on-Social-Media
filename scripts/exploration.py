import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud  

# Charger les données
data = pd.read_csv("data/Tweets.csv")

# Distribution des sentiments
plt.figure(figsize=(10, 6))
sns.countplot(x="airline_sentiment", data=data)
plt.title("Distribution des sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Nombre de tweets")
plt.show()

# Nuage de mots pour les tweets négatifs
negative_tweets = " ".join(data[data["airline_sentiment"] == "negative"]["text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(negative_tweets)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nuage de mots pour les tweets négatifs")
plt.show() 
