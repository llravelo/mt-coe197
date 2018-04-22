#!/usr/bin/env python3

import os

import numpy as np
from keras import Input, Model

from pyfasttext import FastText

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


batch_size = 2*64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 512 # Latent dimensionality of the encoding space.

MAX_SEQUENCE_LENGTH = 32
MAX_NUM_WORDS = 20000
# FastText 300D vectors
EMBEDDING_DIM = 300

def parse_corpora(path):
    texts_eng = []
    texts_fil = []
    for c in os.listdir(path):
        with open(os.path.join(path, c), 'r') as f:
            for line in f:
                try:
                    eng, fil = line.strip().split('\t')
                except ValueError:
                    continue
                # Add start and end of sequence markers
                eng = "'sos' " + eng + " 'eos'"
                # fil = 's_o_s ' + fil + ' e_o_s'
                texts_eng.append(eng)
                texts_fil.append(fil)
    return texts_eng, texts_fil


def convert_to_sequence(texts, max_num_words):
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    # Convert from words to integers
    sequences = tokenizer.texts_to_sequences(texts)
    # Get maximum sequence length
    max_seq_len = max(map(len, sequences))
    # Make sure all sequences have the same length
    data = pad_sequences(sequences, maxlen=max_seq_len)
    return tokenizer.word_index, data


def make_embedding_matrix(word_index, tagalog=True):
    fname = 'cc.tl.300.bin' if tagalog else 'wiki.en.bin'
    model = FastText(os.path.join('embeddings', fname))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_matrix[i] = model.get_numpy_vector(word)
    return embedding_matrix


from keras.layers import LSTM, Dense, Embedding


def main():
    texts_eng, texts_fil = parse_corpora('corpus')

    word_index_eng, data_eng = convert_to_sequence(texts_eng, MAX_NUM_WORDS)
    word_index_fil, data_fil = convert_to_sequence(texts_fil, MAX_NUM_WORDS)

    # Remove sequence markers
    encoder_input_data = data_fil[:, 1:-1]
    encoder_input_data[encoder_input_data == 1] = 0

    # Remove end of sequence markers
    decoder_input_data = data_eng[:, :-1].copy()

    # Remove start of sequence markers
    decoder_target_data = data_eng[:, 1:].copy()
    decoder_target_data[decoder_target_data == 1] = 0

    # print(decoder_input_data)
    # print(decoder_target_data)
    # return

    # return encoder_input_data, decoder_input_data, word_index_eng

    embedding_eng = make_embedding_matrix(word_index_eng, tagalog=False)
    embedding_fil = make_embedding_matrix(word_index_fil, tagalog=True)

    fil_embedding_layer = Embedding(len(word_index_fil) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_fil],
                                    # input_length=encoder_input_data.shape[1],
                                    trainable=False)

    eng_embedding_layer = Embedding(len(word_index_eng) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_eng],
                                    # input_length=decoder_input_data.shape[1],
                                    trainable=False)

    # return fil_embedding_layer, eng_embedding_layer


    latent_dim = 512

    encoder_inputs = Input(shape=(None, ))
    x = fil_embedding_layer(encoder_inputs)
    x = LSTM(latent_dim, return_sequences=True)(x)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    num_decoder_tokens = len(word_index_eng)
    decoder_inputs = Input(shape=(None, ))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_embedded = eng_embedding_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    from keras.utils.vis_utils import plot_model
    from keras.utils import to_categorical

    decoder_target_data = to_categorical(decoder_target_data)[:,:, 1:]

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    plot_model(model)

    # Compile & run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    # Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!


    print(data_eng.shape, data_fil.shape)
    print(encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)


    # model.load_weights('s2s.h5')
    model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)

    # Save model
    model.save('s2s.h5')

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedded, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.summary()

    target_rev_dict = {v: k for k, v in word_index_eng.items()}
    target_rev_dict[0] = '<UNK>'



    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = 1

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            # print(output_tokens)
            # print(output_tokens.shape)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = target_rev_dict[sampled_token_index]
            decoded_sentence += sampled_word + " "

            # Exit condition: either hit max length
            # or find stop character.
            # if sampled_word in [".", "?", "!"] or
            if (sampled_word == '' or
                    len(decoded_sentence) > 33):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return decoded_sentence

    indexes = np.random.randint(0, len(texts_fil), 100)

    for seq_index in indexes:
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', texts_fil[seq_index])
        print('Decoded sentence:', decoded_sentence)


if __name__ == '__main__':
    main()
