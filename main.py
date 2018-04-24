#!/usr/bin/env python3

import numpy as np

from keras.callbacks import ModelCheckpoint

from nmt import data
from nmt.buffering import buffered_gen_threaded as buf
from nmt.models import create_models

# from buffering import buffered_gen_threaded as buf

batch_size = 2*64  # Batch size for training.
epochs = 50 # Number of epochs to train for.
latent_dim = 512 # Latent dimensionality of the encoding space.
embedding_dim = 300


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

    embedding_weights = np.load('embedding-weights.npz')
    e_tl = embedding_weights['tl']
    e_en = embedding_weights['en']
    loader = data.loader(encoder_input_data, decoder_input_data, decoder_target_data,
                         e_tl, e_en, batch_size)

    model = create_models(embedding_dim, latent_dim, embedding_dim)[0]
    model.summary()

    # Compile & run training
    model.compile(optimizer='adagrad', loss='mse')
    # Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!

    checkpoint = ModelCheckpoint('s2s.{epoch:02d}.h5', verbose=True, save_weights_only=True)

    model.fit_generator(buf(loader), len(encoder_input_data)//batch_size, epochs, callbacks=[checkpoint])


if __name__ == '__main__':
    main()
