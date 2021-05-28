"""
Writes csv data by web scraping r/mbti.
Natural language file has the text of comments and the mbti personality dimensions
of the commenter as columns. Bag file replaces the text with a sparse matrix
of tf-idf statistics for each word in the entire comment set for each comment.
Vocab file maps each word to its index in tf-idf.
"""
import re
import numpy as np
import pandas as pd
from pmaw import PushshiftAPI
from sklearn.feature_extraction.text import CountVectorizer
import json

limit = 900000 # The number of comments to search for. Only the comments with an mbti label are ultimately kept.

nl_file_name = 'natural_language.csv'
bag_file_name = 'term_frequencies.csv'
vocab_file_name = 'vocab_tf.json'


def _collect_data():
    """
    Collects comments from r/mbti and returns the data as a DataFrame.
    :return: the dataframe
    """
    df = pd.DataFrame(columns=['text', 'E/I', 'S/N', 'T/F', 'J/P'])
    api = PushshiftAPI()
    comments = api.search_comments(subreddit='mbti', limit=limit, fields=['body', 'author_flair_text'])
    for comment in comments:
        body = comment['body']
        body = body.lower()
        body = body.replace('\n\n', ' ')
        body = re.sub(r'[^\w\s]', '', body)
        flair = comment['author_flair_text']
        if flair is not None and len(flair) == 4:
            df.loc[len(df)] = [body, flair[0], flair[1], flair[2], flair[3]]
    return df


def main():
    """
    Collects the data and writes the data into the csv files.
    """
    df = pd.read_csv(nl_file_name)
    #df = _collect_data()
    #df.to_csv(nl_file_name, index=False)
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(df['text'].apply(lambda x: np.str_(x)))
    #with open(idf_file_name, 'w') as idf_file:
    #    idf_file.write(json.dumps(list(vectorizer.idf_)))
    with open(vocab_file_name, 'w') as vocab_file:
        vocab_file.write(json.dumps(vectorizer.vocabulary_))
    df2 = pd.DataFrame.sparse.from_spmatrix(matrix)
    df2['E/I'] = df['E/I']
    df2['S/N'] = df['S/N']
    df2['T/F'] = df['T/F']
    df2['J/P'] = df['J/P']
    df2.to_csv(bag_file_name, index=False)


if __name__ == '__main__':
    main()