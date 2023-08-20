from transformers import pipeline

def classify_sentiment(text):
    sentiment_classifier = pipeline("sentiment-analysis")
    result = sentiment_classifier(text)

    # Extract the label with the highest confidence
    label = result[0]['label']
    confidence = result[0]['score']

    return label, confidence

# Example usage:
text = "I love this product, it's amazing!"
label, confidence = classify_sentiment(text)
print(f"Sentiment: {label}, Confidence: {confidence:.4f}")