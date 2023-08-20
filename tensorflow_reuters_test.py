import keras
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model  # Import load_model from keras.models
import numpy as np

# Load the saved model
loaded_model = load_model("reuters_model.keras")

# Define the maximum sequence length for text
max_sequence_length = 200

# Load the Reuters dataset and split it into training and testing sets
num_words = 10000  # Limit the number of words in the vocabulary
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words, test_split=0.2)

# Retrieve the word index mapping from the Reuters dataset
word_index = reuters.get_word_index()

# Convert integer sequences to text sequences
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<OOV>"
x_train_text = [' '.join([index_to_word.get(index, '<OOV>') for index in sequence]) for sequence in x_train]
x_test_text = [' '.join([index_to_word.get(index, '<OOV>') for index in sequence]) for sequence in x_test]

# Initialize the tokenizer with a specified vocabulary size
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train_text)

# Define a function to retrieve the Reuters class label from the index
def get_reuters_class_label(index):
    # Define a dictionary mapping class indices to class labels
    class_labels = {
        0: 'cocoa',
        1: 'grain',
        2: 'veg-oil',
        3: 'earn',
        4: 'acq',
        5: 'wheat',
        6: 'copper',
        7: 'housing',
        8: 'money-supply',
        9: 'coffee',
        10: 'sugar',
        11: 'trade',
        12: 'reserves',
        13: 'ship',
        14: 'cotton',
        15: 'carcass',
        16: 'crude',
        17: 'nat-gas',
        18: 'cpi',
        19: 'money-fx',
        20: 'interest',
        21: 'gnp',
        22: 'meal-feed',
        23: 'alum',
        24: 'oilseed',
        25: 'gold',
        26: 'tin',
        27: 'strategic-metal',
        28: 'livestock',
        29: 'retail',
        30: 'ipi',
        31: 'iron-steel',
        32: 'rubber',
        33: 'heat',
        34: 'jobs',
        35: 'lei',
        36: 'bop',
        37: 'zinc',
        38: 'orange',
        39: 'pet-chem',
        40: 'dlr',
        41: 'gas',
        42: 'silver',
        43: 'wpi',
        44: 'hog',
        45: 'lead'
    }

    # Retrieve the class label based on the given index
    if index in class_labels:
        return class_labels[index]
    else:
        return "Class label not found"

# Define a function to predict the Reuters class of a given text
def predict_reuters_class(text, model, token):
    # Tokenize and pad the input text
    sequence = token.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    # Predict the class probabilities
    predi_probabilities = model.predict(padded_sequence)

    # Determine the class with the highest probability
    predi_class = np.argmax(predi_probabilities)

    return predi_class, predi_probabilities

# text_to_predict = "Weak demand from the construction industry has pushed zinc prices to levels which leave some miners with little gain"
text_to_predict = " Chocolate makers like Hershey and Mondelez face tougher trading conditions over the next year as they attempt to pass on soaring cocoa costs to cash-strapped consumers who are cutting back."

predicted_class, predicted_probabilities = predict_reuters_class(text_to_predict, loaded_model, tokenizer)

# Print the predicted class index, predicted class label, and predicted class probabilities
print("Predicted Class Index:", predicted_class)
print("Predicted Class:", get_reuters_class_label(predicted_class))
print("Predicted Probabilities:", predicted_probabilities)
