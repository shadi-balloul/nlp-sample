from nltk import TweetTokenizer
from CleanTweet import tweet_text_cleaner
import pandas as pd


# return the n-grams
def make_ngram(tokens, n):
    # if the length of the tokens is smaller than n-grams
    # we return the right n-grams with n = length of the list of tokens
    if len(tokens) < n:
        return make_ngram(tokens, len(tokens))
    else:
        ngrams = zip(*[tokens[i:] for i in range(n)])
        result = [" ".join(ngram) for ngram in ngrams]
        if len(result) == 0:
            return tokens
        return result


# get the text, apply cleaning, return the n-grams
def process_text(text, stemmer, n):
    tokenizer = TweetTokenizer()
    clean_text = tweet_text_cleaner(text, stemmer)
    # if the we want the 1-grams we return the cleaned text tokenized
    if n == 1:
        return tokenizer.tokenize(clean_text)
    else:
        # return the ngrams of the tokenized cleaned text
        tokens = tokenizer.tokenize(clean_text)
        return make_ngram(tokens, n)


# return the tweets as n-grams and return all the features produced from the data_file
# the training features will be merged in classification and will be the input
# for the classification model
def load_data_and_features(data_file, label, stemmer, n):
    data_features = list()
    data = list()
    # we read the the file
    infile = pd.read_csv(data_file, encoding='utf-8')
    # for each tweet
    for line in infile.Tweet:
        # we process the text and return the requested n-grams as features
        text_features = process_text(line, stemmer, n)
        # the hole features of the file
        data_features += text_features
        # we add the features with the label in the data list
        # the data list will be a list of (features,label)
        data.append((text_features, label))
    return data, data_features


# prepare the features to be added into the classifier
# check each feature in document features if it's contained in corpus_features
# which include all the features
def document_features(document, corpus_features):
    document_words = set(document)
    features = {}
    for word in corpus_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


# return the tweet with the corresponded label
def read_csv(data_file, label, stemmer):
    text_data = list()
    labels = list()
    # we read the the file
    infile = pd.read_csv(data_file, encoding='utf-8')
    # for each tweet
    for line in infile['Tweet']:
        # we clean the tweet
        text_data.append(tweet_text_cleaner(line, stemmer))
        # add the label
        labels.append(label)
    # return the tweets with their labels
    return text_data, labels
