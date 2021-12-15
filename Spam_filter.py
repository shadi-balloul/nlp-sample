import pandas as pd
import re

from nltk import TweetTokenizer

spams = list()
toknizer = TweetTokenizer()

# get all the spams from the spam lexicon
with open('Objects/spam_lexicon.txt', "r", encoding='UTF-8') as s:
    for row in s:
        row = re.sub(r"\n", "", row)
        row = row.strip()
        spams.append(row)

# if the tweet has a word which is a spam
def has_spam(tweet):
    tweet_tokenized = toknizer.tokenize(tweet)
    for word in tweet_tokenized:
        if word in spams:
            return True
    return False


# delete scams from the databases
# we used this function to delete the spam tweets from the our dataset
def filter_tweets(infile, outfile):
    df = pd.read_csv(infile, na_values=[" "])
    index = 0
    spams = 0
    for tweet in df['Tweet']:
        try:
            if has_spam(tweet):
                df = df.drop(df.index[index])
                spams += 1
        except:
            continue
    df.to_csv(outfile, encoding='utf-8-sig', index=False)

