import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vocab_file_name = 'vocab.json'
with open(vocab_file_name, 'rb') as file:
    vocab = json.load(file)

model_file_name = 'Models/100000n_100d_tf/ptype_model.pkl'
with open(model_file_name, 'rb') as file:
    model = pickle.load(file)


def predict(comment):
    count = CountVectorizer()
    tfs = count.fit_transform([comment])
    tfs = tfs.toarray()
    reverse_count_vocab = {index: word for word, index in count.vocabulary_.items()}
    entry = np.zeros(len(vocab))
    for i in range(len(reverse_count_vocab)):
        if reverse_count_vocab[i] in vocab.keys():
            entry[vocab[reverse_count_vocab[i]]] = tfs[0][i]
    print(model.predict([entry]))


def main():
    predict('i like rabbits')


if __name__ == '__main__':
    main()
