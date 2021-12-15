import re
import string
from nltk import TweetTokenizer

import ArabicTranslator

arabic_punctuations = ''''''

# get the arabic punctuations
with open('Objects/Punctuations.txt', 'r', encoding='UTF-8') as f:
    arabic_punctuations += f.readlines()[0]

# get the english punctuations
english_punctuations = string.punctuation

# merge the two lists.
punctuations_list = arabic_punctuations + english_punctuations

# get the stopwords
stop_words = list()
with open('Objects/sw.txt', "r", encoding='UTF-8') as sw:
    for row in sw:
        row = re.sub(r"\n", "", row)
        row = re.sub(r" ", "", row)
        stop_words.append(row)

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

# nltk tokenizer
toknizer = TweetTokenizer()


# normalize arabic text.
def normalize_arabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("لاا", "لا", text)
    text = re.sub("چ", "ك", text)
    text = re.sub("ڤ", "ف", text)
    return text


# we remove all duplicates for further study we can process the redundancy.
# we don't use the translated words option for further study we can use it efficiently.

def remove_duplicates_norm_engwords(text, translate_flag):
    # tokenize the tweet
    list_of_tokens = toknizer.tokenize(text)
    # remove the redundant words and emotes
    words = list(dict.fromkeys(list_of_tokens))
    result = list()
    for word in words:
        # remove the english words
        if re.findall('[a-zA-Z]', word):
            if translate_flag:
                # Translate english words in tweet either: google, textblob, mymemory
                word = ArabicTranslator.translate_english_words(word, 'google')

                # normalize the translated arabic word
                word = normalize_arabic(re.sub(r'(.)\1+', r'\1\1', word))
                if word in stop_words:
                    continue
                else:
                    result.append(word)
            else:
                continue
        else:
            # normalize the tokens
            word = normalize_arabic(re.sub(r'(.)\1+', r'\1\1', word))
            if word in stop_words:
                continue
            else:
                result.append(word)
    # remove the redundant tokens after normalization
    result = list(dict.fromkeys(result))
    return ' '.join(result)


# clean the tweets from URLs, users mentioned, underscore for the hashtags(so it dose not
# merge the 2 words hashtags into one word, we want them as a separate words
# example: Higher_institute -> Higher institute) and removing the punctuations and the arabic diacritics
def remove_special_and_unwanted_chars(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"_", " ", text)
    translator = str.maketrans('', '', punctuations_list)
    text = text.translate(translator)
    return re.sub(arabic_diacritics, '', text)


##########################################
# those 2 function was created for preprocessing the stopwords and when handling the user input
# remove the special characters in the tweet
def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

# remove the arabic diacritics from the tweet
def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text
