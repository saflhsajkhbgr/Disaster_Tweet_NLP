# Disaster_Tweet_NLP

This model uses Word2Vec to embed the words and GRU to train the model

The code requires a pre-trained word2vec weight file named "GoogleNews-vectors-negative300.bin", which you can google and download

The main.py also has NB classifier that takes the problem as an n-gram model, but tested out to be less effective than treating as a sequence

You may find the training data here:
https://www.kaggle.com/competitions/nlp-getting-started/data

I trained the model for about 3 epochs and achieved an accuracy of 0.81
