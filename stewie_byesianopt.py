import numpy as np
import re
import pandas as pd
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from hyperopt import fmin, tpe, hp
from sklearn.model_selection import train_test_split

"""Read the CSV file into a DataFrame and filter it for the company's data."""
df = pd.read_csv('twcs.csv')
company = "AppleSupport"
companyAnswers = df[df['author_id'].str.contains(company)].head(1000)

"""Preprocess tweets and create question-answer pairs."""
questionTweets = companyAnswers.apply(lambda row: re.sub('@[^\s]+|http[^\s]+', '', row['text']), axis=1)
answerTweets = companyAnswers['text']

"""Prepare input and target data."""
input_texts = questionTweets.values
target_texts = ['<START> ' + text + ' <END>' for text in answerTweets.values]

"""Split the data into training and validation sets."""
input_texts_train, input_texts_val, target_texts_train, target_texts_val = train_test_split(
    input_texts, target_texts, test_size=0.2, random_state=42)

"""Tokenize input and target texts."""
input_tokens = set(' '.join(input_texts).split())
target_tokens = set(' '.join(target_texts).split())

"""Create token dictionaries."""
input_token_dict = {token: i for i, token in enumerate(input_tokens)}
target_token_dict = {token: i for i, token in enumerate(target_tokens)}
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

"""Define a special token for unknown words."""
UNK_TOKEN = '<UNK>'
input_token_dict[UNK_TOKEN] = num_encoder_tokens
target_token_dict[UNK_TOKEN] = num_decoder_tokens
num_encoder_tokens += 1
num_decoder_tokens += 1

"""Create one-hot encoded input and target data."""
max_encoder_seq_length = max(len(re.findall(r"[\w']+|[^\s\w]", text)) for text in input_texts)
max_decoder_seq_length = max(len(re.findall(r"[\w']+|[^\s\w]", text)) for text in target_texts)

encoder_input_data = np.zeros((len(input_texts_train), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts_train), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts_train), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts_train, target_texts_train)):
    for t, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_text)):
        token_index = input_token_dict.get(token, input_token_dict[UNK_TOKEN])
        encoder_input_data[i, t, token_index] = 1.
    for t, token in enumerate(target_text.split()):
        token_index = target_token_dict.get(token, target_token_dict[UNK_TOKEN])
        decoder_input_data[i, t, token_index] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, token_index] = 1.


def objective(hyperparameters):
    """Define the objective function for Bayesian optimization."""
    latent_dim, batch_size, epochs = hyperparameters

    """Cast the hyperparameters to integers."""
    latent_dim = int(latent_dim)
    batch_size = int(batch_size)
    epochs = int(epochs)

    """Define the Seq2Seq model."""
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    """Train the model."""
    history = model.fit([encoder_input_data, decoder_input_data],
                        decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

    """Return the validation accuracy (negative for minimization)."""
    return -history.history['val_accuracy'][-1]


"""Define the search space for hyperparameters."""
search_space = [
    hp.uniform('latent_dim', 64, 256),
    hp.uniform('batch_size', 32, 128),
    hp.uniform('epochs', 3, 10)
]

"""Perform Bayesian optimization."""
best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=10)

"""Extract the best hyperparameters."""
best_latent_dim = int(best['latent_dim'])
best_batch_size = int(best['batch_size'])
best_epochs = int(best['epochs'])

print(f"Best Hyperparameters: Latent Dim={best_latent_dim}, Batch Size={best_batch_size}, Epochs={best_epochs}")
