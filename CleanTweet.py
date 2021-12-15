import StripEmoje
from PreProcessing import remove_special_and_unwanted_chars, remove_duplicates_norm_engwords
from Stemmers import stemmer_based_on_number


def tweet_text_cleaner(tweet, stemmer_number):
    try:
        # removing URLs, user_mentions, underscore to use the words in hashtags, the special characters
        # and removing the arabic diacritics
        first_step = remove_special_and_unwanted_chars(tweet)

        # removing the duplicates words in the tweet, the english words, the characters that duplicates in the same
        # word more than 2 times on a row, removing the stop words, normalizing the words.
        second_step = remove_duplicates_norm_engwords(first_step,False)

        # convert emojies into their meaning in english
        third_step = StripEmoje.convert_emojis(second_step)

        # stemming the tweet
        final_step = stemmer_based_on_number(third_step, stemmer_number)

        # return tweet
        return final_step

    except:
        # if there is an error in processing the tweet we simple remove it.
        # by returning an empty string
        return ''


