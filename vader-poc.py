from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sample sentences for sentiment analysis
sentences = [
    "VADER is smart, handsome, and funny.",  # Positive sentence example
    "VADER is smart, handsome, and funny!",  # Punctuation emphasis handled correctly (sentiment intensity adjusted)
    "VADER is very smart, handsome, and funny.",  # Booster words handled correctly (sentiment intensity adjusted)
    "VADER is VERY SMART, handsome, and FUNNY.",  # Emphasis for ALLCAPS handled
    "VADER is VERY SMART, handsome, and FUNNY!!!",  # Combination of signals - VADER appropriately adjusts intensity
    "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!",  # Booster words & punctuation make this close to ceiling for score
    "VADER is not smart, handsome, nor funny.",  # Negation sentence example
    "The book was good.",  # Positive sentence
    "At least it isn't a horrible book.",  # Negated negative sentence with contraction
    "The book was only kind of good.",  # Qualified positive sentence is handled correctly (intensity adjusted)
    "The plot was good, but the characters are uncompelling and the dialog is not great.",  # Mixed negation sentence
    "Today SUX!",  # Negative slang with capitalization emphasis
    "Today only kinda sux! But I'll get by, lol",  # Mixed sentiment example with slang and contrastive conjunction "but"
    "Make sure you :) or :D today!",  # Emoticons handled
    "Catch utf-8 emoji such as such as 💘 and 💋 and 😁",  # Emojis handled
    "Not bad at all"  # Capitalized negation
]

analyzer = SentimentIntensityAnalyzer()

# Analyze and print sentiment scores for each sentence
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))
