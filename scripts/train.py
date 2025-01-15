import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

# Charger les données
data = pd.read_csv("data/Tweets.csv")

# Prétraitement
X = data["text"]
y = data["airline_sentiment"]

# Vectorisation des textes
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Comparaison des modèles
models = {
    "MultinomialNB": MultinomialNB(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} - Accuracy: {accuracy}")

# Sauvegarder le meilleur modèle
best_model_name = max(results, key=results.get)
best_model_name
best_model = models[best_model_name]
joblib.dump(best_model, "models/best_sentiment_analysis_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")