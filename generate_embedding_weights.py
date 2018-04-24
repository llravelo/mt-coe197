#!/usr/bin/env python3

import os.path

import numpy as np
# from pyfasttext import FastText
import fastText

from nmt.data import parse_corpora, convert_to_sequence, MAX_NUM_WORDS, EMBEDDING_DIM

MODEL_EN_BIN = 'wiki.en.bin'
MODEL_TL_BIN = 'cc.tl.300.bin'


def make_embedding_matrix(word_index, fname):
    model = fastText.load_model(os.path.join('nmt/embeddings', fname))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_matrix[i] = fastText.get_word_vector(word)
    return embedding_matrix


def main():
    texts_tl, texts_en = parse_corpora('corpus')
    word_index_en, data_en = convert_to_sequence(texts_en, MAX_NUM_WORDS)
    word_index_tl, data_tl = convert_to_sequence(texts_tl, MAX_NUM_WORDS)
    embedding_tl = make_embedding_matrix(word_index_tl, MODEL_TL_BIN)
    embedding_en = make_embedding_matrix(word_index_en, MODEL_EN_BIN)
    np.savez_compressed('embedding-weights.npz', en=embedding_en, tl=embedding_tl)


if __name__ == '__main__':
    main()
