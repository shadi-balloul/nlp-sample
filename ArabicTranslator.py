import re
import enchant
from textblob.exceptions import NotTranslated
from translate import Translator
from textblob import TextBlob
from googletrans import Translator
from translate.providers import mymemory_translated
import json


# if a word contains english letters
def contains_english_letters(word):
    if re.findall('[a-zA-Z]', word):
        return True
    else:
        return False


# textblob translator
def textblob_translator(word):
    de_blob = TextBlob(word)
    de_blob.correct()
    translated_word = de_blob.translate(from_lang='en', to='ar')
    return translated_word


# using google translate
def google_translator(word):
    translator = Translator()
    return translator.translate(word, dest='ar').text


# using translate library with MyMemory provider
def mymemory_translate(word):
    # we load the credentials for the myMemory provider
    mymemory_cred = json.load(open('Objects/mymemory_cred.json'))
    translator = mymemory_translated.MyMemoryProvider(secret_access_key=mymemory_cred['secret'], to_lang='ar',
                                                      email=mymemory_cred['email'])
    word = translator.get_translation(word)
    return word


# use this function to translate a full tweet into arabic
def translate_english_words(tweet, translator_name):
    try:
        if translator_name == 'google':
            # checks if a word is a valid english word
            d = enchant.Dict("en_US")
            words = re.split(" ", tweet)
            i = 0
            for word in words:
                # checking if the word is suitable English word
                if d.check(word):
                    translated_word = google_translator(word)
                    if word == translated_word:
                        words[i] = ""
                    else:
                        words[i] = translated_word
                    i += 1
                else:
                    if contains_english_letters(word):
                        words[i] = ""
                    else:
                        words[i] = word
                    i += 1
            tweet_translated = ' '.join(words)
            return tweet_translated
        elif translator_name == 'textblob':
            d = enchant.Dict("en_US")
            words = re.split(" ", tweet)
            i = 0
            for word in words:
                if d.check(word):
                    try:
                        translated_word = str(textblob_translator(word))
                        if re.findall('،', translated_word):
                            translations = translated_word.split('،')
                            words[i] = translations[0]
                        else:
                            words[i] = translated_word
                    except NotTranslated:
                        words[i] = ""
                    i += 1
                else:
                    if contains_english_letters(word):
                        words[i] = ""
                    else:
                        words[i] = word
                    i += 1
            tweet_translated = ' '.join(words)
            return tweet_translated

        else:
            d = enchant.Dict("en_US")
            words = re.split(" ", tweet)
            i = 0
            for word in words:
                try:
                    if d.check(word):
                        translated_word = mymemory_translate(word)
                        if re.findall(',', translated_word):
                            translations = translated_word.split(',')
                            words[i] = translations[0]
                            i += 1
                        elif word == translated_word:
                            words[i] = ""
                        else:
                            words[i] = translated_word
                            i += 1
                    else:
                        if contains_english_letters(word):
                            words[i] = ""
                        elif word == " ":
                            words[i] = ""
                        else:
                            words[i] = word
                        i += 1
                except ValueError:
                    words[i] = ""
                    i += 1
            tweet_translated = ' '.join(words)
            return tweet_translated

    except:
        return tweet
