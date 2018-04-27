#!/usr/bin/env python3

import sys

from pyfasttext import FastText

from nmt.models import create_models
from nmt.data import SOS

import numpy as np

from test import decode_sequence


def main():
    model, encoder_model, decoder_model = create_models(300, 512, 300)
    model.load_weights(sys.argv[1])

    ft_en = FastText('embeddings/wiki.en.bin')
    ft_tl = FastText('embeddings/wiki.tl.bin')

    start_seq = ft_en.get_numpy_vector(SOS, normalized=True).reshape(1, 1, -1)

    chars = '.,?!()'

    while True:
        input_sentence = input('Input Tagalog: ').lower()#'kamusta ka ?'
        
        for c in chars:
            input_sentence = input_sentence.replace(c, ' ' + c + ' ')

        print('Embedding...')
        input_seq = input_sentence.lower().split()
        aaa = np.zeros((1,15,300), dtype='float32')
        for i, w in enumerate(input_seq):
            aaa[0, i] = ft_tl.get_numpy_vector(w, normalized=True)
        #input_seq = [ft_tl.get_numpy_vector(i, normalized=True) for i in input_seq]
        #input_seq = np.stack(input_seq).reshape(1, -1, 300)
        input_seq = aaa
        print(input_seq)

        print('Translating...')

        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, ft_en, start_seq)
        print('-')
        print('Input sentence:', input_sentence)
        print('Decoded sentence:', decoded_sentence)
    

if __name__ == '__main__':
    main()
