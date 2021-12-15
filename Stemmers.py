from nltk import SnowballStemmer, TweetTokenizer
from nltk.stem.isri import ISRIStemmer

from ArabicStemmers.arabic_processing_cog_stemmer.arabic_processing_cog.stemming import \
    Light10stemmer as arabic_processing_cog_stemmer

from tashaphyne.stemming import ArabicLightStemmer
import os
import re
import random

toknizer = TweetTokenizer()


def cog_stemmer(tweet):
    stemmed_tweet = list()
    for token in toknizer.tokenize(tweet):
        stem_word = arabic_processing_cog_stemmer.stem_token(token)
        stemmed_tweet.append(stem_word)
    return ' '.join(stemmed_tweet)


def snowball_stemmer(tweet):
    stemmer = SnowballStemmer('arabic')
    stemmed_tweet = list()

    for word in toknizer.tokenize(tweet):
        stemmed_word = stemmer.stem(word.lower())
        stemmed_tweet.append(stemmed_word)
    return ' '.join(stemmed_tweet)


def isri_stemmer(tweet):
    stemmed_tweet = list()
    isri_stemmer = ISRIStemmer()

    for word in toknizer.tokenize(tweet):
        # stem word
        stem_word = isri_stemmer.stem(word)
        # add new stem to dict
        stemmed_tweet.append(stem_word)

    return ' '.join(stemmed_tweet)


def improved_isri_stemmer(tweet):
    stemmed_tweet = list()
    stemmer = ISRIStemmer()
    for word in toknizer.tokenize(tweet):
        word = stemmer.norm(word, num=1)  # remove diacritics which representing Arabic short vowels
        word = stemmer.pre32(word)  # remove length three and length two prefixes in this order
        word = stemmer.suf32(word)  # remove length three and length two suffixes in this order
        word = stemmer.waw(word)  # remove connective ‘و’ if it precedes a word beginning with ‘و’
        word = stemmer.norm(word, num=2)  # normalize initial hamza to bare alif
        stemmed_tweet.append(word)
    return ' '.join(stemmed_tweet)


def khoja_stemmer(Tweet):
    try:
        path = 'ArabicStemmers\\shereen_khoja_stemmer\\Khoja'

        # random number to be named for the temp file
        r = random.getrandbits(50)
        in_text_path = os.path.join(path, 'TempFiles\\' + str(r) + 'in.txt')
        out_text_path = os.path.join(path, 'TempFiles\\' + str(r) + 'out.txt')
        # the path for the in and out temp files
        f = open(out_text_path, "w+")
        # create the output file
        f.close()

        with open(in_text_path, 'w+', encoding='utf-8') as f:
            # write the tweet in the in file
            f.write('%s\n' % Tweet)
        # khoja path
        khoja_path = os.path.join(path, 'khoja-stemmer-command-line.jar')

        # call the jar file for the in and out files
        os.system('java -jar ' + khoja_path + ' ' + in_text_path + ' ' + out_text_path)

        output_stemmed_tweets = []
        with open(out_text_path, 'r', encoding='utf-8') as f:
            for listitem in f:
                # listitem = re.sub(" ","",listitem)
                listitem = re.sub('\n', "", listitem)
                output_stemmed_tweets.append(listitem.strip())

        # remove the in and our temp files
        os.remove(in_text_path)
        os.remove(out_text_path)
        return ' '.join(output_stemmed_tweets)
    except:
        # the error that happens is that when we are using the prallel loading for the data
        # so more than one call for that khoja jar which leads to an error which is more than one process
        # are calling the jar file
        # so we rerun the function for that tweet
        return khoja_stemmer(Tweet)


def tashaphyne_stemmer(tweet):
    stemmed_tweet = list()
    for word in toknizer.tokenize(tweet):
        ArListem = ArabicLightStemmer()
        stemmed_tweet.append(ArListem.light_stem(word))
    return ' '.join(stemmed_tweet)


# stem the tweet based on the stemmer number
# on error we recall this function which is more likely to happen
# for the khoja stemmer which we handled before.
def stemmer_based_on_number(tweet, stemmer_number):
    try:
        if stemmer_number == 1:
            return cog_stemmer(tweet)
        elif stemmer_number == 2:
            return snowball_stemmer(tweet)
        elif stemmer_number == 3:
            return isri_stemmer(tweet)
        elif stemmer_number == 4:
            return improved_isri_stemmer(tweet)
        elif stemmer_number == 5:
            return khoja_stemmer(tweet)
        else:
            return tashaphyne_stemmer(tweet)
    except:
        return stemmer_based_on_number(tweet, stemmer_number)
