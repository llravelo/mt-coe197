#!/usr/bin/env python3

from keras import Input, Model
from keras.layers import LSTM, Dense


def create_models(input_dim, latent_dim, output_dim, output_activation=None):
    encoder_inputs = Input(shape=(None, input_dim))
    x = LSTM(latent_dim, return_sequences=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, output_dim))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs, _, _ = decoder_lstm2(decoder_outputs)
    decoder_dense = Dense(output_dim, activation=output_activation)
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_state_input_h2 = Input(shape=(latent_dim,))
    decoder_state_input_c2 = Input(shape=(latent_dim,))
    decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_outputs, state_h2, state_c2 = decoder_lstm2(decoder_outputs, initial_state=decoder_states_inputs2)
    decoder_states = [state_h, state_c, state_h2, state_c2]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs + decoder_states_inputs2,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model
