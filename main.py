#!/usr/bin/env python3

import numpy as np

from keras import Input, Model
from keras.callbacks import ModelCheckpoint

from nmt import data

batch_size = 2*64  # Batch size for training.
epochs = 50 # Number of epochs to train for.
latent_dim = 512 # Latent dimensionality of the encoding space.


from keras.layers import LSTM, Dense, Embedding

from nmt.buffering import buffered_gen_threaded as buf

# from buffering import buffered_gen_threaded as buf

def main():
    texts_tl, texts_en = data.parse_corpora('corpus')
    word_index_tl, word_index_en, encoder_input_data, decoder_input_data, decoder_target_data = data.preprocess(texts_en, texts_tl)

    print(texts_tl[0])
    print(encoder_input_data[0])

    print(texts_en[0])
    print(decoder_input_data[0])
    print(decoder_target_data[0])

    print('Number of samples:', len(texts_tl))
    print('Number of unique input tokens:', len(word_index_tl))
    print('Number of unique output tokens:', len(word_index_en))
    print('Max sequence length for inputs:', encoder_input_data.shape[1])
    print('Max sequence length for outputs:', decoder_input_data.shape[1])

    loader = data.loader(encoder_input_data, decoder_input_data, decoder_target_data, batch_size)

    embedding_weights = np.load('embedding-weights.npz')


    embedding_tl = Embedding(len(word_index_tl) + 1,
                                    data.EMBEDDING_DIM,
                                    weights=[embedding_weights['tl']],
                                    trainable=False)

    embedding_en = Embedding(len(word_index_en) + 1,
                                    data.EMBEDDING_DIM,
                                    weights=[embedding_weights['en']],
                                    trainable=False)

    latent_dim = 512

    encoder_inputs = Input(shape=(None, ))
    x = embedding_tl(encoder_inputs)
    x = LSTM(latent_dim, return_sequences=True)(x)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    num_decoder_tokens = len(word_index_en)
    decoder_inputs = Input(shape=(None, ))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_embedded = embedding_en(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    from keras.utils.vis_utils import plot_model


    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    plot_model(model)

    # Compile & run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    # Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!

    print(encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)

    checkpoint = ModelCheckpoint('s2s.{epoch:02d}.h5', verbose=True, save_weights_only=True)

    model.fit_generator(buf(loader), len(encoder_input_data)//batch_size, epochs, callbacks=[checkpoint])
    # model.load_weights('s2s.02.h5')
    #
    # # model.load_weights('s2s.h5')
    # model.fit([encoder_input_data, decoder_input_data],
    #           decoder_target_data,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_split=0.1)

    # Save model
    # model.save('s2s.h5')

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

    target_rev_dict = {v: k for k, v in word_index_en.items()}
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

    indexes = np.random.randint(0, len(texts_tl), 100)

    for seq_index in indexes:
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', texts_tl[seq_index])
        print('Decoded sentence:', decoded_sentence)


if __name__ == '__main__':
    main()
