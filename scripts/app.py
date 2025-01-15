from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the best model and vectorizor    
model = joblib.load("models/best_sentiment_analysis_model.pkl") 
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True) 

