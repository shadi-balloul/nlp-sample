import re
import emoji
from nltk.tokenize import TweetTokenizer
import json

# emojies_to_words is a dictionary contains each emojie with the corresponded meaning in english
emojies_to_words = json.load(open("Objects/emojis.json"))
tokenizer = TweetTokenizer()


def remove_extra_whitespaces(text):
    s = ' '.join(text.split())
    return s


# get the emojis from a text
def extract_emojis(s):
    return re.findall(r'[^\w\s,]', s)


# first way is to translate is to convert emojie to their corresponding meaning in english or translate the
# into arabic and add them to the tweet text
def convert_emojis(tweet):
    try:
        # remove the extra white spaces from the tweet
        tweet = remove_extra_whitespaces(tweet)
        # tokenize the tweet
        list_of_tokens = tokenizer.tokenize(tweet)
        # extract the emojies
        emojie_in_text = extract_emojis(tweet)
        i = 0
        # if there is some emojies in the tweet
        if len(emojie_in_text) > 0:
            # for each token in list of tokens
            for token in list_of_tokens:
                # if the token is in the emojies
                if list_of_tokens[i] in emojie_in_text:
                    # if the emoji is in the lexicon
                    if list_of_tokens[i] in emojies_to_words:
                        # we wont remove the underscore for making each emoji a single word
                        list_of_tokens[i] = emojies_to_words[list_of_tokens[i]].replace("-", "_").replace("_&", "")
                    # if the emojie is not in the dictionary we use the emoji package to return the meaning
                    else:
                        if emoji.demojize(list_of_tokens[i]) != list_of_tokens[i]:
                            list_of_tokens[i] = emoji.demojize(list_of_tokens[i]).replace("-", "_").replace("_&", "")
                        else:
                            list_of_tokens[i] = ""
                i += 1
            return ' '.join(list_of_tokens)
        else:
            return tweet
    except:
        return tweet
