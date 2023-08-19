from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from mastodon_connector import search_tag

# Retrieve a list of posts related to "bitcoin" using the mastodon_connector module
list_posts = search_tag("bitcoin")

# Sentiment Analysis with VADER
analyzer = SentimentIntensityAnalyzer()

print("Sentiment Analysis with VADER:")
for sentence in list_posts:
    # Limit the displayed sentence to the first 40 characters for readability
    truncated_sentence = sentence[0:40]
    
    # Perform sentiment analysis using VADER
    vs = analyzer.polarity_scores(sentence)
    
    # Print the truncated sentence and VADER sentiment scores
    print("{:-<65} {}".format(truncated_sentence, str(vs)))

# Sentiment Analysis with TextBlob
print("\nSentiment Analysis with TextBlob:")
for sentence in list_posts:
    # Limit the displayed sentence to the first 40 characters for readability
    truncated_sentence = sentence[0:40]
    
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(sentence)
    
    # Print the truncated sentence and TextBlob sentiment
    print("{:-<65} {}".format(truncated_sentence, str(blob.sentiment)))
