"""
This program is for analyzing the data collected by collector.py. This program trains machine learning
models and plots statistics about the data.
"""
import json
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler
import pickle
import nltk


folder = 'Models/'
data_folder = 'Data_Plots/'
nl_file_name = '1000natural_language.csv'
vocab = None
types = ['ESTJ', 'ESTP', 'ESFJ', 'ESFP', 'ENTJ', 'ENTP', 'ENFJ', 'ENFP',
         'ISTJ', 'ISTP', 'ISFJ', 'ISFP', 'INTJ', 'INTP', 'INFJ', 'INFP']
target_word = 'i'
train_size = 10000  # If memory is not an issue, there are 31270 comments available,
                    # reasonably set train_size to 25270 and test size to 6000
test_size = 2500
stop_words = ['estj', 'estp', 'esfj', 'esfp', 'entj', 'entp', 'enfj', 'enfp',
              'istj', 'istp', 'isfj', 'isfp', 'intj', 'intp', 'infj', 'infp',
              'estjs', 'estps', 'esfjs', 'esfps', 'entjs', 'entps', 'enfjs', 'enfps',
              'istjs', 'istps', 'isfjs', 'isfps', 'intjs', 'intps', 'infjs', 'infps',
              'fi', 'fe', 'si', 'se', 'ne', 'ni', 'ti', 'te',
              'fis', 'fes', 'sis', 'ses', 'nes', 'nis', 'tis', 'tes',
              'extravert', 'extraverts', 'extroversion', 'introvert', 'introverts', 'introversion',
              'intuitives', 'sensing', 'intuitive', 'thinking', 'feeling', 'perceiving', 'judging',
              'fise', 'seni', 'tini', 'nite', 'nidom', 'sidom', 'tine',
              'fises', 'senis', 'tinis', 'nites', 'nidoms', 'sidoms', 'tines']
parts_of_speech_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
                   'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN',
                   'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
nlp_features = ['word_count', 'letters', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
                'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
                'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'E_I',
                'S_N', 'T_F', 'J_P', 'ptype']


def get_words(s):
    return len(str(s).split())


def avg_words(nl):
    """
    Adds a column to the natural language data for word count of each comment.
    Plots the average number of words for each dimension in one plot and each
    type in another plot
    :param nl: the natural language data
    """
    print('getting average words per comment')
    nl['word_count'] = nl['text'].apply(get_words)
    e_i = nl.groupby('E_I')['word_count'].mean()
    s_n = nl.groupby('S_N')['word_count'].mean()
    t_f = nl.groupby('T_F')['word_count'].mean()
    j_p = nl.groupby('J_P')['word_count'].mean()
    ptype = nl.groupby('ptype')['word_count'].mean()

    df = pd.DataFrame({'dim': ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P'], 'avg': [
        e_i.loc['E',], e_i.loc['I',], s_n.loc['S',], s_n.loc['N',],
        t_f.loc['T',], t_f.loc['F',], j_p.loc['J',], j_p.loc['P',]]})
    df2 = pd.DataFrame({'ptype': types,
                        'avg': [ptype.loc[types[i]] for i in range(16)]})

    sns.catplot(data=df, x='dim', y='avg', kind='bar')
    plt.title('Average Words per Comment by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Average Words')
    plt.savefig(data_folder + 'avg_words_by_dim.png', bbox_inches='tight')

    sns.catplot(data=df2, x='ptype', y='avg', kind='bar')
    plt.title('Average Words per Comment by Type')
    plt.xlabel('Type')
    plt.ylabel('Average Words')
    plt.xticks(rotation=45)
    plt.savefig(data_folder + 'avg_words_by_type.png', bbox_inches='tight')


def get_appearances(s):
    """
    Returns the number of times the global target word appears in the
    input string.
    :param s: the string to search through
    :return: The number of appearances of the word
    """
    count = 0
    for word in str(s).split():
        if word == target_word:
            count = count + 1
    return count


def word_frequency(nl):
    """
    Adds a column to the natural language data for the word count and the
    number of appearances of the global target word. Plots the frequency of
    the word for each type in one plot and each dimension in another plot.
    :param nl: the natural language data
    """
    print('getting word frequency')
    nl['word_count'] = nl['text'].apply(get_words)
    e_i_total = nl.groupby('E_I')['word_count'].sum()
    s_n_total = nl.groupby('S_N')['word_count'].sum()
    t_f_total = nl.groupby('T_F')['word_count'].sum()
    j_p_total = nl.groupby('J_P')['word_count'].sum()
    ptype_total = nl.groupby('ptype')['word_count'].sum()

    nl[target_word] = nl['text'].apply(get_appearances)
    e_i = nl.groupby('E_I')[target_word].sum().divide(e_i_total)
    s_n = nl.groupby('S_N')[target_word].sum().divide(s_n_total)
    t_f = nl.groupby('T_F')[target_word].sum().divide(t_f_total)
    j_p = nl.groupby('J_P')[target_word].sum().divide(j_p_total)
    ptype = nl.groupby('ptype')[target_word].sum().divide(ptype_total)

    df = pd.DataFrame({'dim': ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P'], 'avg': [
        e_i.loc['E',], e_i.loc['I',], s_n.loc['S',], s_n.loc['N',],
        t_f.loc['T',], t_f.loc['F',], j_p.loc['J',], j_p.loc['P',]]})
    df2 = pd.DataFrame({'ptype': types,
                        'avg': [ptype.loc[types[i]] for i in range(16)]})

    sns.catplot(data=df, x='dim', y='avg', kind='bar')
    plt.title('Frequency of "' + target_word + '" by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Frequency')
    plt.savefig(data_folder + target_word + '_dim.png', bbox_inches='tight')

    sns.catplot(data=df2, x='ptype', y='avg', kind='bar')
    plt.title('Frequency of "' + target_word + '" by Type')
    plt.xlabel('Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.savefig(data_folder + target_word + '_type.png', bbox_inches='tight')


def count_types(nl):
    """
    Plots the number of comments of each type
    :param nl: the natural language data
    """
    print('counting types')
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
    plt.savefig(data_folder + 'count_type.png', bbox_inches='tight')


def get_letters(s):
    """
    Returns the number of letters in a string, ignoring spaces
    :param s: The string to search through
    :return: the number of letters in the string
    """
    return len(str(s).replace(' ', ''))


def avg_word_length(nl):
    """
    Plots the average word length comments of each type and dimension. Also adds
    the word count, number of letters, and average word length for each comment
    as features.
    :param nl: the natural language data
    """
    print('collecting average word length')
    nl['word_count'] = nl['text'].apply(get_words)
    e_i_words = nl.groupby('E_I')['word_count'].mean()
    s_n_words = nl.groupby('S_N')['word_count'].mean()
    t_f_words = nl.groupby('T_F')['word_count'].mean()
    j_p_words = nl.groupby('J_P')['word_count'].mean()
    ptype_words = nl.groupby('ptype')['word_count'].mean()

    nl['letters'] = nl['text'].apply(get_letters)
    e_i = nl.groupby('E_I')['letters'].mean().divide(e_i_words)
    s_n = nl.groupby('S_N')['letters'].mean().divide(s_n_words)
    t_f = nl.groupby('T_F')['letters'].mean().divide(t_f_words)
    j_p = nl.groupby('J_P')['letters'].mean().divide(j_p_words)
    ptype = nl.groupby('ptype')['letters'].mean().divide(ptype_words)

    nl['word_length'] = nl['letters'].divide(nl['word_count'])  # for ml features only

    df = pd.DataFrame({'dim': ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P'], 'avg': [
        e_i.loc['E', ], e_i.loc['I', ], s_n.loc['S', ], s_n.loc['N', ],
        t_f.loc['T', ], t_f.loc['F', ], j_p.loc['J', ], j_p.loc['P', ]]})
    df2 = pd.DataFrame({'ptype': types,
                        'avg': [ptype.loc[types[i]] for i in range(16)]})

    sns.catplot(data=df, x='dim', y='avg', kind='bar')
    plt.title('Average Word Length by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Length')
    plt.savefig(data_folder + 'word_length_by_dim.png', bbox_inches='tight')

    sns.catplot(data=df2, x='ptype', y='avg', kind='bar')
    plt.title('Average Word Length by Type')
    plt.xlabel('Type')
    plt.ylabel('Length')
    plt.xticks(rotation=45)
    plt.savefig(data_folder + 'word_length_by_type.png', bbox_inches='tight')


def avg_letters_per_comment(nl):
    """
    Plots the average number of letters per comment for each dimension in one plot
    and each type in another plot
    :param nl: the natural language data
    """
    print('collecting average letters per comment')
    nl['letters'] = nl['text'].apply(get_letters)
    e_i = nl.groupby('E_I')['letters'].mean()
    s_n = nl.groupby('S_N')['letters'].mean()
    t_f = nl.groupby('T_F')['letters'].mean()
    j_p = nl.groupby('J_P')['letters'].mean()
    ptype = nl.groupby('ptype')['letters'].mean()

    df = pd.DataFrame({'dim': ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P'], 'avg': [
        e_i.loc['E', ], e_i.loc['I', ], s_n.loc['S', ], s_n.loc['N', ],
        t_f.loc['T', ], t_f.loc['F', ], j_p.loc['J', ], j_p.loc['P', ]]})
    df2 = pd.DataFrame({'ptype': types,
                        'avg': [ptype.loc[types[i]] for i in range(16)]})

    sns.catplot(data=df, x='dim', y='avg', kind='bar')
    plt.title('Average Letters per Comment by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Average Letters')
    plt.savefig(data_folder + 'avg_letters_by_dim.png', bbox_inches='tight')

    sns.catplot(data=df2, x='ptype', y='avg', kind='bar')
    plt.title('Average Letters per Comment by Type')
    plt.xlabel('Type')
    plt.ylabel('Average Letters')
    plt.xticks(rotation=45)
    plt.savefig(data_folder + 'avg_letters_by_type.png', bbox_inches='tight')


def get_parts_of_speech(s):
    """
    Returns the parts of speech of each word in the input string as a list of
    tuples (word, part of speech)
    :param s: The string to search through
    :return: the parts of speech of each word in the input string as a list of
    tuples (word, part of speech)
    """
    return nltk.pos_tag(nltk.word_tokenize(str(s)))


def frequency_single_part_of_speech(tagged_tokens, part):
    """
    Takes a list of tuples (word, part of speech) and the part of speech to look
    for and returns the frequency of that part of speech
    :param tagged_tokens: list of tuples (word, part of speech)
    :param part: a part of speech in nltk format
    """
    if len(tagged_tokens) == 0:
        return 0
    count = 0
    for token in tagged_tokens:
        if token[1] == part:
            count = count + 1
    return count / len(tagged_tokens)


def parts_of_speech(nl):
    """
    Adds a column for each part of speech to the natural language data
    for the uses of that part of speech in the comment. Plots the frequency
    of uses of each part of speech for each dimension and type
    :param nl: The natural language data.
    """
    print('collecting parts of speech')
    nl['temp'] = nl['text'].apply(get_parts_of_speech)
    dims = ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']

    for type in types:
        dims.append(type)
    df = pd.DataFrame(dtype=int, columns=dims, index=parts_of_speech_list)
    for part in parts_of_speech_list:
        nl[part] = nl['temp'].apply(lambda x: frequency_single_part_of_speech(x, part))
        df.loc[part, 'E'] = nl.groupby('E_I')[part].mean().loc['E']
        df.loc[part, 'I'] = nl.groupby('E_I')[part].mean().loc['I']
        df.loc[part, 'S'] = nl.groupby('S_N')[part].mean().loc['S']
        df.loc[part, 'N'] = nl.groupby('S_N')[part].mean().loc['N']
        df.loc[part, 'T'] = nl.groupby('T_F')[part].mean().loc['T']
        df.loc[part, 'F'] = nl.groupby('T_F')[part].mean().loc['F']
        df.loc[part, 'J'] = nl.groupby('J_P')[part].mean().loc['J']
        df.loc[part, 'P'] = nl.groupby('J_P')[part].mean().loc['P']
        for ptype in types:
            df.loc[part, ptype] = nl.groupby('ptype')[part].mean().loc[ptype]
    nl = nl.drop(columns='temp')

    readable = ['Coordinating Conjunction', 'Cardinal Digit', 'Determiner',
                'Existential There', 'Foreign Word',
                'Preposition/Subordinating Conjunction', 'Adjective',
                'Adjective, Comparative', 'Adjective, Superlative',
                'List Marker', 'Modal', 'Noun, Singular', 'Noun, Plural',
                'Proper Noun, Singular', 'Proper Noun, Plural', 'Predeterminer',
                'Possessive Ending', 'Personal Pronoun', 'Possessive Pronoun',
                'Adverb', 'Adverb, Comparative',
                'Adverb, Superlative', 'Particle', 'To', 'Interjection',
                'Verb, Base Form', 'Verb, Past Tense',
                'Verb, Gerund/ Present Participle', 'Verb, Past Participle',
                'Verb, Singular Present', 'Verb, 3rd Person Present',
                'wh-determiner (which)', 'wh-pronoun (who, what)',
                'possessive wh-pronoun (whose)', 'wh-adverb (where, when)']
    i = 0
    for part in parts_of_speech_list:
        plt.figure(figsize=(40, 40))
        plot_data = pd.DataFrame({part: df.loc[part, ], 'dim': dims})
        sns.catplot(data=plot_data, x='dim', y=part, kind='bar')
        plt.xlabel('Type/Dimension')
        plt.ylabel('Frequency')
        plt.title('Frequency of ' + readable[i] + ' Part of Speech by Type and Dimension', y=1.08)
        plt.xticks(rotation=45)
        plt.savefig(data_folder + part.replace(' ', '') + '_pos.png', bbox_inches='tight')
        i = i + 1


def plot_features():
    """
    Plots the feature importances
    """
    print('plotting features')
    data = json.load(open(folder + 'E_I_RFResults.json'))
    importances = data[2][0:20]
    features = [feature for feature, score in importances]
    scores = [score for feature, score in importances]
    df = pd.DataFrame({'feature': features, 'score': scores})
    plt.figure(figsize=(20, 20), dpi=300)
    sns.catplot(data=df, x='feature', y='score', kind='bar')
    plt.title('Top 20 Predictors of Extroverted/Introverted')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, fontsize=7)
    plt.savefig(data_folder + 'E_I_features.png', bbox_inches='tight')

    data = json.load(open(folder + 'S_N_RFResults.json'))
    importances = data[2][0:20]
    features = [feature for feature, score in importances]
    scores = [score for feature, score in importances]
    df = pd.DataFrame({'feature': features, 'score': scores})
    plt.figure(figsize=(20, 20), dpi=300)
    sns.catplot(data=df, x='feature', y='score', kind='bar')
    plt.title('Top 20 Predictors of Sensing/Intuitive')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, fontsize=7)
    plt.savefig(data_folder + 'S_N_features.png', bbox_inches='tight')

    data = json.load(open(folder + 'T_F_RFResults.json'))
    importances = data[2][0:20]
    features = [feature for feature, score in importances]
    scores = [score for feature, score in importances]
    df = pd.DataFrame({'feature': features, 'score': scores})
    plt.figure(figsize=(20, 20), dpi=300)
    sns.catplot(data=df, x='feature', y='score', kind='bar')
    plt.title('Top 20 Predictors of Thinking/Feeling')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, fontsize=7)
    plt.savefig(data_folder + 'T_F_features.png', bbox_inches='tight')

    data = json.load(open(folder + 'J_P_RFResults.json'))
    importances = data[2][0:20]
    features = [feature for feature, score in importances]
    scores = [score for feature, score in importances]
    df = pd.DataFrame({'feature': features, 'score': scores})
    plt.figure(figsize=(20, 20), dpi=300)
    sns.catplot(data=df, x='feature', y='score', kind='bar')
    plt.title('Top 20 Predictors of Judging/Perceiving')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, fontsize=7)
    plt.savefig(data_folder + 'J_P_features.png', bbox_inches='tight')

    data = json.load(open(folder + 'ptype_RFResults.json'))
    importances = data[2][0:20]
    features = [feature for feature, score in importances]
    scores = [score for feature, score in importances]
    df = pd.DataFrame({'feature': features, 'score': scores})
    plt.figure(figsize=(20, 20), dpi=300)
    sns.catplot(data=df, x='feature', y='score', kind='bar')
    plt.title('Top 20 Predictors of Personality Type')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, fontsize=7)
    plt.savefig(data_folder + 'type_features.png', bbox_inches='tight')


def plot_accuracy():
    """
    Plots the testing accuracy for each model.
    """
    print('running plot_accuracy')
    e_i_rf_accuracy = json.load(open(folder + 'E_I_RFResults.json'))[0]
    e_i_lr_accuracy = json.load(open(folder + 'E_I_LRResults.json'))[0]
    s_n_rf_accuracy = json.load(open(folder + 'S_N_RFResults.json'))[0]
    s_n_lr_accuracy = json.load(open(folder + 'S_N_LRResults.json'))[0]
    t_f_rf_accuracy = json.load(open(folder + 'T_F_RFResults.json'))[0]
    t_f_lr_accuracy = json.load(open(folder + 'T_F_LRResults.json'))[0]
    j_p_rf_accuracy = json.load(open(folder + 'J_P_RFResults.json'))[0]
    j_p_lr_accuracy = json.load(open(folder + 'J_P_LRResults.json'))[0]
    ptype_rf_accuracy = json.load(open(folder + 'ptype_RFResults.json'))[0]
    ptype_lr_accuracy = json.load(open(folder + 'ptype_LRResults.json'))[0]
    df = pd.DataFrame({'model': ['E/I Random Forest', 'E/I Logistic Regression',
                                  'S/N Random Forest', 'S/N Logistic Regression',
                                  'T/F Random Forest', 'T/F Logistic Regression',
                                  'J/P Random Forest', 'J/P Logistic Regression',
                                  'Type Random Forest', 'Type Logistic Regression'],
                       'score': [e_i_rf_accuracy, e_i_lr_accuracy,
                                 s_n_rf_accuracy, s_n_lr_accuracy,
                                 t_f_rf_accuracy, t_f_lr_accuracy,
                                 j_p_rf_accuracy, j_p_lr_accuracy,
                                 ptype_rf_accuracy, ptype_lr_accuracy]
                       })
    sns.catplot(data=df, x='model', y='score', kind='bar')
    plt.title('Test Accuracy of Each Model')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    for i in range(10):
        plt.text(i, df['score'][i], df['score'][i])
    plt.savefig(data_folder + 'models_accuracy.png', bbox_inches='tight')


def important_words(importances):
    """
    Translates feature_importances_ into a list words as text sorted by their importance
    :param importances: the importances given by feature_importances_ of the model
    :return: a sorted list of tuples of word, importance
    """
    reverse_vocab = {index: word for word, index in vocab.items()}
    i = len(reverse_vocab)
    for feature in nlp_features:
        reverse_vocab[i] = feature
        i = i + 1
    important_words = [(reverse_vocab[index], score) for index, score in enumerate(importances) if
                       index in reverse_vocab.keys()]
    important_words.sort(key=itemgetter(1), reverse=True)
    return important_words


def plot_tree(model, dim):
    """
    Plots the tree for a model of a dimension
    :param model: The model
    :param dim: The dimension
    """
    if dim == 'ptype':
        ascending_labels = types.copy()
        ascending_labels.sort()
    else:
        ascending_labels = [dim[0], dim[2]]
        ascending_labels.sort()
    vocab_copy = [(word, index) for word, index in vocab.items()]
    vocab_copy.sort(key=itemgetter(1))
    words = [word for word, index in vocab_copy]
    for feature in nlp_features:
        words.append(feature)
    plt.figure(figsize=(20, 20), dpi=300)
    tree.plot_tree(model, feature_names=words, class_names=ascending_labels, filled=True)
    plt.savefig(folder + dim + '_' + 'tree.png', bbox_inches='tight')


def balance_labels(features_train, labels_train, dim):
    """
    Upsamples the training data based on the dimension
    :param features_train: the features training data
    :param labels_train: the labels for the training data
    :param dim: the dimension
    :return: the upsampled data
    """
    bag = pd.concat([features_train, labels_train], axis=1)
    largest = bag[dim].value_counts().max()
    unique = bag[dim].unique()
    recombined = pd.DataFrame(columns=bag.columns)
    for value in unique:
        if isinstance(value, str):
            recombined = pd.concat([recombined, bag[bag[dim] == value].sample(n=largest, replace=True)])
    recombined.dropna()
    return recombined.drop(columns=['ptype', 'E_I', 'S_N', 'T_F', 'J_P']), recombined[
        ['ptype', 'E_I', 'S_N', 'T_F', 'J_P']]


def split_data(nl):
    """
    Gets the training and test features and labels from the natural language file
    :return: a tuple of training features, training labels, test features, test labels. Labels include all possible
    labels, so they must be split into singular columns for actual use in the model
    """
    print('splitting data')
    train_nl = nl.head(train_size)
    train_vectorizer = CountVectorizer(stop_words=stop_words)
    matrix = train_vectorizer.fit_transform(train_nl['text'].apply(lambda x: np.str_(x)))
    train_bag = pd.DataFrame.sparse.from_spmatrix(matrix)
    train_bag = train_bag.join(train_nl[nlp_features])
    labels_train = train_bag[['ptype', 'E_I', 'S_N', 'T_F', 'J_P']]
    features_train = train_bag.drop(columns=['ptype', 'E_I', 'S_N', 'T_F', 'J_P'])
    print('training data built')

    global vocab
    vocab = train_vectorizer.vocabulary_  # test data has to use the word indices of the training data, making this code complex

    test_nl = nl.tail(test_size).reset_index()
    test_vectorizer = CountVectorizer(vocabulary=vocab, stop_words=stop_words)
    test_matrix = test_vectorizer.fit_transform(test_nl['text'].apply(lambda x: np.str_(x)))
    test_bag = pd.DataFrame.sparse.from_spmatrix(test_matrix)
    test_bag = test_bag.join(test_nl[nlp_features])
    labels_test = test_bag[['ptype', 'E_I', 'S_N', 'T_F', 'J_P']]
    features_test = test_bag.drop(columns=['ptype', 'E_I', 'S_N', 'T_F', 'J_P'])
    print('test data built')

    return features_train, labels_train, features_test, labels_test


def model_dim(dim, features_train, labels_train, features_test, labels_test, depth, split, leaf, features):
    """
    Selects the dimension out of the labels to train and test a model on. Writes a file for
    the model, a tree of the model, and the results of test accuracy, training accuracy, and
    most important words
    :param dim: the dimension to select to model
    :param features_train: the features of the training data
    :param labels_train: labels for all dimensions of the training data
    :param features_test: the features of the training data
    :param labels_test: labels for all dimensions of the labels data
    """
    print(dim)
    features_train = MaxAbsScaler().fit_transform(features_train)
    features_test = MaxAbsScaler().fit_transform(features_test)
    labels_train = labels_train[dim]
    labels_test = labels_test[dim]
    print('RF modeling')
    RF_model = RandomForestClassifier(n_estimators=1, max_depth=depth, min_samples_split=split,
                                      min_samples_leaf=leaf, max_features=features, random_state=0)
    RF_model.fit(features_train, labels_train)
    print('RF saving')
    with open(folder + dim + '_' + 'RFresults.json', 'w') as file:
        predictions = RF_model.predict(features_test)
        json.dump((accuracy_score(labels_test, predictions),
                   accuracy_score(labels_train, RF_model.predict(features_train)),
                   important_words(RF_model.feature_importances_)), file)
    with open(folder + dim + '_' + 'RFmodel.pkl', 'wb') as file:
        pickle.dump(RF_model, file)
    plot_tree(RF_model.estimators_[0], dim)

    print('LR modeling')
    LR_model = LogisticRegression(max_iter=500, random_state=42, solver='saga')
    LR_model.fit(features_train, labels_train)
    print('LR saving')

    with open(folder + dim + '_' + 'LRresults.json', 'w') as file:
        predictions = LR_model.predict(features_test)
        json.dump((accuracy_score(labels_test, predictions),
                   accuracy_score(labels_train, LR_model.predict(features_train))), file)

    with open(folder + dim + '_' + 'LRmodel.pkl', 'wb') as file:
        pickle.dump(LR_model, file)


def make_models(features_train, labels_train, features_test, labels_test):
    """
    Trains models for each dimension
    :param features_train: the features of the training data
    :param labels_train: labels for all dimensions of the training data
    :param features_test: the features of the training data
    :param labels_test: labels for all dimensions of the labels data
    """
    model_dim('E_I', features_train, labels_train, features_test, labels_test,
              depth=15, split=40, leaf=20, features='auto')
    model_dim('S_N', features_train, labels_train, features_test, labels_test,
              depth=15, split=40, leaf=20, features='auto')
    model_dim('T_F', features_train, labels_train, features_test, labels_test,
              depth=15, split=40, leaf=20, features='auto')
    model_dim('J_P', features_train, labels_train, features_test, labels_test,
              depth=15, split=40, leaf=20, features='auto')
    model_dim('ptype', features_train, labels_train, features_test, labels_test,
              depth=15, split=40, leaf=20, features='auto')


def main():
    nl = pd.read_csv(nl_file_name, dtype='str')
    word_frequency(nl)
    parts_of_speech(nl)
    avg_words(nl)
    count_types(nl)
    avg_word_length(nl)
    avg_letters_per_comment(nl)
    features_train, labels_train, features_test, labels_test = split_data(nl)
    make_models(features_train, labels_train, features_test, labels_test)
    plot_accuracy()
    plot_features()


if __name__ == '__main__':
    main()
