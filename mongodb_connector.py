from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from mastodon_connector import search_tag
import pymongo

# Connect to MongoDB (replace with your MongoDB connection details)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]  # Replace "mydatabase" with your database name
collection = db["sentiment_results"]  # Replace "sentiment_results" with your collection name

list_posts = search_tag("bitcoin")

# Vader
analyzer = SentimentIntensityAnalyzer()
for sentence in list_posts:
    vs = analyzer.polarity_scores(sentence)
    # Save the sentiment analysis result to MongoDB
    collection.insert_one({
        "sentence": sentence,
        "sentiment": "VADER",
        "polarity_scores": vs
    })

# TextBlob
for sentence in list_posts:
    blob = TextBlob(sentence)
    # Save the sentiment analysis result to MongoDB
    collection.insert_one({
        "sentence": sentence,
        "sentiment": "TextBlob",
        "polarity_scores": blob.sentiment.polarity
    })

# Close the MongoDB connection
client.close()
