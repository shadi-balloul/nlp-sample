from tweepy import OAuthHandler, API
import json

# the twitter API tokens we need them to access the website
twitter_keys = json.load(open("Objects/Twitter_keys.json"))


def getTwitterAPI():
    # Authentication with twitter using the tokens
    auth = OAuthHandler(twitter_keys['consumer_key'], twitter_keys['consumer_secret'])
    auth.set_access_token(twitter_keys['access_token'], twitter_keys['access_token_secret'])
    api = API(auth)
    return api

