import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, SimpleRNN, GRU, Dense

text1 =  "I went to the park the other day with my parents and I (complete with 5 words)" 
text2 = "I hurt my foot the other while playing football with my friends in the park, but I still wanted to keep playing as I was having a lot of fun. Then I came home and I ate a lot of carbonara that I had prepared yesterday so now my belly hurts and so does my (complete with 1 word)"
text3 = "Basketball is (complete with 50 words)"

# Upload CSV
df = pd.read_csv("Random_English_Sentences.csv")
corpus = df["text"].astype(str).tolist()

# Words to numbers
tokenizer = Tokenizer()
token = tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Sentences to numbers
sentences = []
for line in corpus:
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        sentences.append(tokens[:i+1])

# Entry and exit of the model
x = sentences[:, :-1]
last_words = []
for seq in sentences:
    last_words.append(seq[-1])

y = np.zeros((len(last_words), total_words))
for i, w in enumerate(last_words):
    y[i, w] = 1



# calculating the lenght
lenght = []
for s in sentences:
    lenght.append(len(s))

max_len = max(lenght)
# All sentences the same lenght
sentences = pad_sequences(
    sentences,
    maxlong = max_len,
    padding = "pre"
)

# Models:
def models(type):
    model = Sequential()
    model.add(
        Embedding(
            total_words, 32, input_length= max_len-1
        )
    )
    if type == "RNN":
        model.add(SimpleRNN(32))
    elif type == "LSTM":
        model.add(LSTM(32))
    elif type == "GRU":
        model.add(GRU(32))

    model.add(Dense(total_words, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

# Generate text
def generate(model, text, n):
    for _ in range(n):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len-1, padding="pre")
        pred =np.argmax(model.predict(sequence, verbose=0))

        for word, idx in tokenizer.word_index.items():
            if idx == pred:
                text += " " + word
                break
    return text

# Train models:
models_dict = {}
for type in ["RNN", "LSTM", "GRU"]:
    print("Training model:", type)
    model = models(type)
    model.fit(x, y, epochs=20, verbose=0)
    models_dict[type] = model

for type in models_dict:
    print("1) ", generate(models_dict[type], text1, 5))
    print("2) ", generate(models_dict[type], text2, 1))
    print("3) ", generate(models_dict[type], text1, 50))
