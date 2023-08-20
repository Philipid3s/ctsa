from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pymongo

app = Flask(__name__)

# Connect to MongoDB (replace with your MongoDB connection details)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]  # Replace "mydatabase" with your database name
collection = db["sentiment_results"]  # Replace "sentiment_results" with your collection name

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']

    # Vader
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    vader_result = {
        "sentence": text,
        "sentiment": "VADER",
        "polarity_scores": vs
    }

    # TextBlob
    blob = TextBlob(text)
    textblob_result = {
        "sentence": text,
        "sentiment": "TextBlob",
        "polarity_scores": blob.sentiment.polarity
    }

    # Save results to MongoDB
    collection.insert_many([vader_result, textblob_result])

    return jsonify({'vader_sentiment': vs, 'textblob_sentiment': blob.sentiment.polarity})

if __name__ == '__main__':
    app.run(debug=True)
