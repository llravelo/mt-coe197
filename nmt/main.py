#!/usr/bin/env python3

import os

import numpy as np

from pyfasttext import FastText

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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
                eng = '\a ' + eng + ' \b'
                fil = '\a ' + fil + ' \b'
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
    fname = 'cc.tl.300.bin' if tagalog else 'wiki-news-300d-1M.bin'
    model = FastText(os.path.join('embeddings', fname))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_matrix[i] = model.get_numpy_vector(word)


def main():
    texts_eng, texts_fil = parse_corpora('corpus')

    word_index_eng, data_eng = convert_to_sequence(texts_eng, MAX_NUM_WORDS)
    word_index_fil, data_fil = convert_to_sequence(texts_fil, MAX_NUM_WORDS)

    embedding_eng = make_embedding_matrix(word_index_eng, tagalog=False)
    embedding_fil = make_embedding_matrix(word_index_fil, tagalog=True)

    print(embedding_eng.shape)
    print(embedding_fil.shape)


if __name__ == '__main__':
    main()
