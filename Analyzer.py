import json
from operator import itemgetter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle


nl_file_name = 'natural_language.csv'
bag_file_name = 'term_frequencies.csv'
vocab_file_name = 'vocab.json'
types = ['ESTJ', 'ESTP', 'ESFJ', 'ESFP', 'ENTJ', 'ENTP', 'ENFJ', 'ENFP',
          'ISTJ', 'ISTP', 'ISFJ', 'ISFP', 'INTJ', 'INTP', 'INFJ', 'INFP']
n = 100000


def get_words(s):
    return len(str(s).split())


def plot_avg_words(nl):
    nl = nl.copy()
    nl['count'] = nl['text'].apply(get_words)
    nl['type'] = nl['E/I'].astype(str) + nl['S/N'].astype(str) + nl['T/F'].astype(str) + nl['J/P']
    e_i = nl.groupby('E/I')['count'].mean()
    s_n = nl.groupby('S/N')['count'].mean()
    t_f = nl.groupby('T/F')['count'].mean()
    j_p = nl.groupby('J/P')['count'].mean()
    ptype = nl.groupby('type')['count'].mean()

    df = pd.DataFrame({'dim': ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P'], 'avg': [
         e_i.loc['E', ], e_i.loc['I', ], s_n.loc['S', ], s_n.loc['N', ],
         t_f.loc['T', ], t_f.loc['F', ], j_p.loc['J', ], j_p.loc['P', ]]})
    df2 = pd.DataFrame({'ptype': types,
                        'avg': [ptype.loc[types[i]] for i in range(16)]})

    sns.catplot(data=df, x='dim', y='avg', kind='bar')
    plt.title('Average Words per Comment by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Average Words')
    plt.savefig('Data_Plots/avg_words_by_dim.png', bbox_inches='tight')

    sns.catplot(data=df2, x='ptype', y='avg', kind='bar')
    plt.title('Average Words per Comment by Type')
    plt.xlabel('Type')
    plt.ylabel('Average Words')
    plt.xticks(rotation=45)
    plt.savefig('Data_Plots/avg_words_by_type.png', bbox_inches='tight')


def plot_count_types(nl):
    nl = nl.copy()
    nl['ptype'] = nl['E/I'].astype(str) + nl['S/N'].astype(str) + nl['T/F'].astype(str) + nl['J/P']
    counts = nl['ptype'].value_counts(sort=False)
    df = pd.DataFrame({'ptype': types,
                        'count': [counts[ptype] for ptype in types]})
    sns.catplot(data=df, x='ptype', y='count', kind='bar')
    plt.title('Count of Each Type')
    for i in range(16):
        plt.text(i, df['count'][i], df['count'][i])
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('Data_Plots/count_type.png', bbox_inches='tight')


def important_words(importances, vocab):
    important_words = [(word, importances[index]) for word, index in vocab.items() if index < len(importances)]
    important_words.sort(key=itemgetter(1), reverse=True)
    return important_words


def plot_tree(model, dim, vocab, depth):
    if dim == 'ptype':
        ascending_labels = types.copy()
        ascending_labels.sort()
    else:
        ascending_labels = [dim[0], dim[2]]
        ascending_labels.sort()
    vocab = vocab.copy()
    vocab = [(word, index) for word, index in vocab.items()]
    vocab.sort(key=itemgetter(1))
    vocab = [word for word, index in vocab]
    plt.figure(figsize=(50, 50), dpi=300)
    tree.plot_tree(model, feature_names=vocab, class_names=ascending_labels, filled=True)
    plt.savefig('Models/' + str(n) + 'n_' + str(depth) + 'd_tf/' + dim + '_' + 'tree.png', bbox_inches='tight')


def equal_labels(bag, dim):
    if (dim != 'ptype'):
        first_dim = bag[bag[dim] == dim[0]]
        second_dim = bag[bag[dim] == dim[2]]
        n = min(len(first_dim), len(second_dim))
        bag = first_dim.head(n).append(second_dim.head(n))
        bag = bag.sample(frac=1)
        return bag
    else:
        smallest_type = 300000
        for ptype in types:
            if len(bag[bag['ptype'] == ptype]) < smallest_type:
                smallest_type = len(bag[bag['ptype'] == ptype])
        evendf = pd.DataFrame()
        for ptype in types:
            evendf = evendf.append(bag[bag['ptype'] == ptype].head(smallest_type))
        evendf = evendf.sample(frac=1)
        return evendf


def model_dim(bag, vocab, dim, depth):
    bag['ptype'] = bag['E/I'].astype(str) + bag['S/N'].astype(str) + bag['T/F'].astype(str) + bag['J/P']
    bag = equal_labels(bag, dim)
    labels = bag[dim].astype(str)
    features = bag.drop(columns=['ptype', 'E/I', 'S/N', 'T/F', 'J/P'])
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    print(features_train)
    print(labels_train)
    print(features_test)
    print(labels_test)
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    if dim != 'ptype':
        dim = dim[0] + '_' + dim[2]
    with open('Models/' + str(n) + 'n_' + str(depth) + 'd_tf/' + dim + '_' + 'results.json', 'w') as file:
        file.write(json.dumps((accuracy_score(labels_test, model.predict(features_test)),
                               accuracy_score(labels_train, model.predict(features_train)),
                               important_words(model.feature_importances_, vocab))))
    with open('Models/' + str(n) + 'n_' + str(depth) + 'd_tf/' + dim + '_' + 'model.pkl', 'wb') as file:
        pickle.dump(model, file)
    plot_tree(model, dim, vocab, depth)



def make_models(bag, depth):
    copy = bag.copy()
    f = open(vocab_file_name)
    vocab = json.load(f)
    model_dim(bag, vocab, 'E/I', depth)
    bag = copy
    model_dim(bag, vocab, 'S/N', depth)
    bag = copy
    model_dim(bag, vocab, 'T/F', depth)
    bag = copy
    model_dim(bag, vocab, 'J/P', depth)
    bag = copy
    model_dim(bag, vocab, 'ptype', depth)


def main():
    nl = pd.read_csv(nl_file_name)
    plot_avg_words(nl)
    plot_count_types(nl)
    bag = pd.read_csv(bag_file_name, nrows=n)
    #for depth in range(75, 101, 5):
    depth = 'no_max'
    make_models(bag, depth)


if __name__ == '__main__':
    main()