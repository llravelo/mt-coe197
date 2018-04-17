# Machine Translation Using Sequence to Sequence Models
Members:
1. Leiko Ravelo
2. Paolo Valdez
3. Darwin Bautista

### Presentation Materials
1. [Google slides](https://docs.google.com/presentation/d/1mjo4LcduXuh5jWLy6pzUe-ZqpJvtBWL_9-tAxVEPq2g/edit?usp=sharing)
2. LaTeX project is in Slack chat

### Development Environment
Preferably work on a virtual environment (Python 3).
See [this guide](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/) for installing virtual environments.

I also thought it would be a good idea to use jupyter notebooks for this.

Packages (so far):
* Tensorflow
* Keras
* Jupyter
* Ipython

```
pip install keras tensorflow jupyter ipython
```

The following command will open a browser to the jupyter environment.
```
jupyter notebook
```

It's also possible to run a jupyter notebook remotely. See [running a notebook server](http://jupyter-notebook.readthedocs.io/en/stable/public_server.html). (Can access through vpn w/ opera)

### Useful Resources
Add here some resources you think might be useful for the project.
* Hyperparameter optimization: [hyperopt](https://github.com/hyperopt/hyperopt)
* Save and Load Keras Models: [link](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)
* Stanford CS224n NLP with DL: [link](http://web.stanford.edu/class/cs224n/syllabus.html)
* Hyperparamter optimization guide: [link](https://www.jeremyjordan.me/hyper-parameter-tuning/)
* Machine Translation Best Practices "mini guide": [link](http://ruder.io/deep-learning-nlp-best-practices/index.html#neuralmachinetranslation)

Keras Tutorials:
* Francois Chollet DL with Python: [link](https://github.com/fchollet/deep-learning-with-python-notebooks)
* ML-AI experiments: [link](https://github.com/kmsravindra/ML-AI-experiments)
* NMT-Keras: [link](https://nmt-keras.readthedocs.io/en/latest/)

### Research Papers
Add here relevant research papers

### Tasks
#### Dataset Creation
Once we have the dataset available, curate it and format it properly (tab indented text file).
#### Data Preprocessing
Convert text data from file to acceptable representation. May also need to remove punctuations (like comma, period, must be settled early on). Theres one-hot representation, which is pretty easy to implement. word2vec and glove is also available but may take some time to implement.

Relevant Resources:
* [Chris Albon Tutorials](https://chrisalbon.com/#machine_learning)
* [One-hot representation tutorial](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-one-hot-encoding-of-words-or-characters.ipynb)
* [word embeddings tutorial](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.1-using-word-embeddings.ipynb)
* [pretrained glove representation](https://nlp.stanford.edu/projects/glove/)
* [fasttext](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md)
#### Model Design and Metric checking
Objective of the project is to create an optimal working machine translator. Define objectives and a success metric. There's BLEU score but let's wait for sir's definition. Standard RNN architecture ('Vanilla Model') used for machine translation can be seen on stanford lectures.

> I think it's also important to decouple training and actual translation. Need a way to save the model and load it on a different python file. -Leiko

Relevant Resources:
* [Basic Neural Machine Translator (Eng-Fr)](https://github.com/kmsravindra/ML-AI-experiments/blob/master/AI/Neural%20Machine%20Translation/Neural%20machine%20translation%20-%20Encoder-Decoder%20seq2seq%20model.ipynb)

#### Hyperparameter Tuning
Possible hyperparameters include. Task is to identify which hyperparameters are important, and find generally accepted ranges. One solution is do a random search through the hyperparameter space to find the model that gives optimal results.

I have found [hyperopt](https://github.com/hyperopt/hyperopt) although I'm not sure how it works yet.

Possible hyperparameters include:
1. Learning rate
2. Gradient Descent Optimizer (SGD, Adam, RMSprop)
3. Minibatch size
4. Epochs
4. RNN layers (1-4)
5. Choice of RNN architecture (GRU, RNN, LSTM)
6. Misc (Attention model, bidirectional lstm, deep lstm)

>I'm not sure yet which ones of the above are important -Leiko
