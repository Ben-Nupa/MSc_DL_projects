import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop

'''Useful variables'''
PATH_TO_DATA = 'data'
GLOVE_DIR = 'glove.6B'  # Directory for pre-trained embedding weights
MAX_SEQUENCE_LENGTH = 40
MAX_NUM_WORDS = 18000
EMBEDDING_DIM = 100
OUTPUT_FOLDER = "results"
SAVE_PREDICTIONS = True


def load_sst_data(folder: str) -> tuple:
    """Extracts the SST data from the given folder (sentences and labels from each set)."""

    def extract_sentences(filename: str, is_testing_set=False) -> tuple:
        """Extracts the sentences and labels from the given file from the SST dataset."""
        with open(filename, 'r', encoding='utf-8') as file:
            sentences = []
            labels = []
            for line in file.readlines():
                if is_testing_set:  # No label in testing set
                    sentences.append(line)
                else:
                    labels.append(int(line[0]))
                    sentences.append(line[2:])
        return sentences, labels

    tr_sentences, tr_labels = extract_sentences(os.path.join(folder, 'stsa.fine.train'))
    val_sentences, val_labels = extract_sentences(os.path.join(folder, 'stsa.fine.dev'))
    te_sentences, _ = extract_sentences(os.path.join(folder, 'stsa.fine.test.X'), is_testing_set=True)

    return tr_sentences, tr_labels, val_sentences, val_labels, te_sentences


def plot_training_history(training_history):
    """Plots the learning curve."""
    n_epochs = len(training_history.history['acc'])
    # Plot training
    abscissa = 1 + np.arange(n_epochs)
    # Loss
    plt.figure()
    plt.plot(abscissa, training_history.history['loss'], '.', label='Tr loss')
    plt.plot(abscissa, training_history.history['val_loss'], '.', label='Val loss')
    plt.legend()

    # Accuracy
    plt.figure()
    plt.plot(abscissa, training_history.history['acc'], '.', label='Tr acc')
    plt.plot(abscissa, training_history.history['val_acc'], '.', label='Val acc')
    plt.legend()


'''Preparing the data'''
# Load text (function defined at the beginning of part 4)
tr_sentences, y_tr, val_sentences, y_val, te_sentences = load_sst_data(os.path.join(PATH_TO_DATA, 'SST'))

# Tokenize and build data matrices
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(tr_sentences)
tr_sequences = tokenizer.texts_to_sequences(tr_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)
te_sequences = tokenizer.texts_to_sequences(te_sentences)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_tr = pad_sequences(tr_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_val = pad_sequences(val_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_te = pad_sequences(te_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print('Shape of training data tensor:', X_tr.shape)

'''
Preparing the embedding layer.
Download the GloVe word embeddings from : http://nlp.stanford.edu/data/glove.6B.zip
'''
# Build the index mapping
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# Build the embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Build the embedding layer
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

'''Build the model'''
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(2)(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# x = Conv1D(64, 5, activation='relu')(x)
# x = MaxPooling1D(2)(x)
# x = BatchNormalization()(x)
# x = Dropout(0.5)(x)

x = LSTM(128, dropout=0.5, recurrent_dropout=0.5)(x)
x = BatchNormalization()(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
preds = Dense(5, activation='softmax')(x)

model = Model(sequence_input, preds)

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])
print(model.summary())

# Fit
history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=40, batch_size=128)
plot_training_history(history)

# Predict and save
if SAVE_PREDICTIONS:
    y_predicted = model.predict(X_te).argmax(axis=-1)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    with open(os.path.join(OUTPUT_FOLDER, 'ConvFC_lstm_y_test_sst.txt'), 'w') as file:
        for i in y_predicted:
            file.write(str(i) + '\n')


plt.show()
