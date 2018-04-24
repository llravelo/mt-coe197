#!/usr/bin/env python3

import os
import sys

import numpy as np

from pyfasttext import FastText

from nmt import data
from nmt.models import create_models


def decode_sequence(input_seq, encoder_model, decoder_model, ft_model):
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
        sampled_word = ft_model.words_for_vector(output_tokens[0, -1, :], k=1)[0][0]
        print(sampled_word)
        decoded_sentence += sampled_word + " "

        # Exit condition: either hit max length
        # or find stop character.
        # if sampled_word in [".", "?", "!"] or
        if (sampled_word == data.EOS or
                len(decoded_sentence) > 100):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = output_tokens

        # Update states
        states_value = [h, c]

    return decoded_sentence


def main():
    texts_tl, texts_en = data.parse_corpora('corpus')
    # word_index_tl, word_index_en, encoder_input_data, decoder_input_data, decoder_target_data = data.preprocess(
    #     texts_en, texts_tl)
    #


    embedding_dim = 300
    latent_dim = 512
    model, encoder_model, decoder_model = create_models(embedding_dim, latent_dim, embedding_dim)
    model.load_weights(sys.argv[1])

    indexes = np.random.randint(0, len(texts_tl), 100)

    ft_model = FastText(os.path.join('embeddings', 'wiki.en.bin'))

    for seq_index in indexes:
        # Take one sequence (part of the training set)
        # for trying out decoding.
        sentence = texts_tl[seq_index]
        input_seq = sentence.split()[1:-1]
        print(input_seq)
        input_seq = np.stack(list(map(ft_model.get_numpy_vector, t)))
        print(input_seq)
        print(input_seq.shape)
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, ft_model)
        print('-')
        print('Input sentence:', texts_tl[seq_index])
        print('Decoded sentence:', decoded_sentence)



if __name__ == '__main__':
    main()