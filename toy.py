#!/usr/bin/env python3

import numpy as np
from keras.utils import to_categorical

from nmt.models import create_models

def main():
    model, encoder, decoder = create_models(10, 128, 13, 'softmax')
    source = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    dst_in = np.array([[11, 9], [11, 8], [11, 7], [11, 6], [11, 5], [11, 4], [11, 3], [11, 2], [11, 1], [11, 0]])
    dst_target = np.array([[9, 12], [8, 12], [7, 12], [6, 12], [5, 12], [4, 12], [3, 12], [2, 12], [1, 12], [0, 12]])

    print(source.shape)
    source = to_categorical(source).reshape(10, 1, -1)
    dst_in = to_categorical(dst_in, 13)
    dst_target = to_categorical(dst_target, 13)

    print(source.shape)
    print(dst_in.shape)
    print(dst_target.shape)

    model.summary()

    model.compile('adagrad', 'categorical_crossentropy')
    model.fit([source, dst_in], [dst_target], epochs=140)

    input = to_categorical(np.array([5]), 10).reshape(1, 1, -1)
    print(input.shape)
    states = encoder.predict(input) #+ [np.zeros((1, 128)), np.zeros((1, 128))]
    # print(states[0].shape)

    token = to_categorical(np.array([11]), 13).reshape(1, 1, -1)

    while True:
        out, h, c = decoder.predict([token] + states)
        i = np.argmax(out)
        print(i)
        if i == 12:
            break
        token = out
        states = [h, c ]






if __name__ == '__main__':
    main()
