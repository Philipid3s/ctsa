import keras
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences


# Load the Reuters dataset
num_words = 10000  # Limit the number of words in the vocabulary
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words, test_split=0.2)
word_index = reuters.get_word_index()

# Convert integer sequences to text
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<OOV>"
x_train_text = [' '.join([index_to_word.get(index, '<OOV>') for index in sequence]) for sequence in x_train]
x_test_text = [' '.join([index_to_word.get(index, '<OOV>') for index in sequence]) for sequence in x_test]

# Tokenize and pad the text sequences
max_sequence_length = 200  # Limit the sequence length
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train_text)
x_train_seq = tokenizer.texts_to_sequences(x_train_text)
x_test_seq = tokenizer.texts_to_sequences(x_test_text)
x_train_pad = pad_sequences(x_train_seq, maxlen=max_sequence_length)
x_test_pad = pad_sequences(x_test_seq, maxlen=max_sequence_length)

# Convert labels to one-hot encoding
num_classes = max(y_train) + 1
print('Classes: ', num_classes)
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

# Build a Sequential model
model = Sequential()
model.add(Embedding(num_words, 128, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 12
history = model.fit(x_train_pad, y_train_onehot, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model
score = model.evaluate(x_test_pad, y_test_onehot, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("reuters_model.keras")
