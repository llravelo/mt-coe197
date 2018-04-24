#!/usr/bin/env python3

import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

SOS = "'sos'"
EOS = "'eos'"
# FastText 300D vectors
EMBEDDING_DIM = 300

MAX_NUM_WORDS = 20000


def parse_corpora(path):
    texts_en = []
    texts_tl = []
    for c in os.listdir(path):
        fname = os.path.join(path, c)
        if not os.path.isfile(fname):
            continue
        with open(fname, 'r') as f:
            for line in f:
                try:
                    en, tl = line.strip().split('\t')
                except ValueError:
                    continue
                # Add start and end of sequence markers
                en = SOS + ' ' + en + ' ' + EOS
                tl = SOS + ' ' + tl + ' ' + EOS
                texts_en.append(en)
                texts_tl.append(tl)
                if len(tl) > 900:
                    print(c)
                    print(tl)
    return texts_tl, texts_en


def convert_to_sequence(texts, max_num_words):
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    # Convert from words to integers
    sequences = tokenizer.texts_to_sequences(texts)
    # Get maximum sequence length
    max_seq_len = max(map(len, sequences))
    # Make sure all sequences have the same length
    data = pad_sequences(sequences, maxlen=max_seq_len, padding='post')
    return tokenizer.word_index, data


def loader(encoder_input_data, decoder_input_data, decoder_target_data, in_embedding, out_embedding, batch_size, shuffle=True):
    num_batches = len(encoder_input_data) // batch_size
    # print(len(encoder_input_data), batch_size)
    num_classes = np.max(decoder_target_data) + 1
    while True:
        if shuffle:
            p = np.random.permutation(len(encoder_input_data))
            encoder_input_data = encoder_input_data[p]
            decoder_input_data = decoder_input_data[p]
            decoder_target_data = decoder_target_data[p]
        enc_in_b = np.array_split(encoder_input_data, num_batches)
        dec_in_b = np.array_split(decoder_input_data, num_batches)
        dec_out_b = np.array_split(decoder_target_data, num_batches)
        while enc_in_b:
            ei = enc_in_b.pop()
            di = dec_in_b.pop()
            do = dec_out_b.pop()
            ei = np.take(in_embedding, ei, axis=0)
            di = np.take(out_embedding, di, axis=0)
            do = np.take(out_embedding, do, axis=0)
            yield [ei, di], [do]


def preprocess(texts_en, texts_tl):
    word_index_en, data_en = convert_to_sequence(texts_en, MAX_NUM_WORDS)
    word_index_tl, data_tl = convert_to_sequence(texts_tl, MAX_NUM_WORDS)

    # Remove sequence markers
    encoder_input_data = data_tl[:, 1:-1]
    encoder_input_data[encoder_input_data == 2] = 0

    # Remove end of sequence markers
    decoder_input_data = data_en[:, :-1].copy()
    decoder_input_data[decoder_input_data == 2] = 0

    # Remove start of sequence markers
    decoder_target_data = data_en[:, 1:].copy()
    # decoder_target_data[decoder_target_data == 1] = 0

    return word_index_tl, word_index_en, encoder_input_data, decoder_input_data, decoder_target_data
