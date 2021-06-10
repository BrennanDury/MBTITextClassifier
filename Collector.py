"""
Writes csv data by web scraping r/mbti.
Natural language file has the text of comments and the mbti personality dimensions
of the commenter as columns. Bag file replaces the text with a sparse matrix
of tf-idf statistics for each word in the entire comment set for each comment.
Vocab file maps each word to its index in tf-idf.
"""
import re
import pandas as pd
from pmaw import PushshiftAPI
import datetime
from datetime import timedelta
import csv


nl_file_name = 'natural_language.csv'
days = 365


def over(s):
    return len(str(s)) > 1000


def collect_data():
    """
    Collects comments from r/mbti and returns the data as a DataFrame.
    :return: the dataframe
    """
    api = PushshiftAPI()
    f = open(nl_file_name, 'w', newline='')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['text', 'E_I', 'S_N', 'T_F', 'J_P', 'ptype'])
    f.close()
    comments = []
    total_collected = 0
    before = datetime.datetime(2021, 6, 3, 0, 0)
    hours_dif = 12
    after = before - timedelta(hours=hours_dif)
    while before > datetime.datetime(2010, 12, 30, 0, 0):  # the creation of the subreddit
        response = api.search_comments(subreddit='mbti', before=int(before.timestamp()), after=int(after.timestamp()),
                                       fields=['body', 'author_flair_text'])
        before = before - timedelta(hours=hours_dif)
        comments.extend(response)
        new_comments = len(comments) - total_collected
        total_collected = len(comments)
        if new_comments != 0:
            hours_dif = (hours_dif + (hours_dif * 800 / new_comments)) / 2  # Go halfway to the estimate
            # to get 800 on next search. If over 1000, comments will be missed. If under 1000, collection
            # is less efficient
            for comment in comments[-new_comments:]: #write each comment as its collected for
                body = comment['body']
                body = body.lower()
                body = body.replace('\n\n', ' ')
                body = re.sub(r'[^\w\s]', '', body)
                flair = comment['author_flair_text']
                if flair is not None and len(flair) == 4:
                    g = open(nl_file_name, 'a')
                    writer = csv.writer(g, delimiter=',')
                    writer.writerow(pd.Series([body, flair[0], flair[1], flair[2], flair[3], flair]))
                    g.close()
        print(before)
        print(hours_dif)
        after = before - timedelta(hours=hours_dif)


def filter_1000():
    df = pd.read_csv(nl_file_name)
    filter = df['text'].apply(over)
    df['text'] = df[filter]
    df.to_csv('1000' + nl_file_name, index=False)

def main():
    collect_data()


if __name__ == '__main__':
    main()
