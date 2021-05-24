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
from sklearn.feature_extraction.text import TfidfVectorizer
import json

limit = 100000 # The number of comments to search for. Only the comments with an mbti label are ultimately kept.

nl_file_name = 'Data/natural_language.csv'
bag_file_name = 'Data/bag_of_words.csv'
vocab_file_name = 'Data/vocab.json'


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
    df = _collect_data()
    df.to_csv(nl_file_name)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(df['text'].apply(lambda x: np.str_(x)))
    df2 = pd.DataFrame.sparse.from_spmatrix(matrix)
    df2['E/I'] = df['E/I']
    df2['S/N'] = df['S/N']
    df2['T/F'] = df['T/F']
    df2['J/P'] = df['J/P']
    with open(vocab_file_name, 'w') as vocab_file:
        vocab_file.write(json.dumps(vectorizer.vocabulary_))
    df2.to_csv(bag_file_name)


if __name__ == '__main__':
    main()