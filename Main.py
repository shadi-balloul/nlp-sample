import glob
import json
import re
import time
import sys
import webbrowser

import joblib
import ConnectToES
import warnings
import tweepy

import CleanTweet
import GUI
import TweetAuth
import os
from Spam_filter import has_spam

# ignore the warnings.
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# first item is all the tweets received and the second item is the tweets sent to elasticsearch
tweets = [0, 0]


# load the default track list to be added in streaming.
def GetTrackList():
    with open("Objects/TrackList.txt", "r", encoding='utf-8') as track_list_file:
        TrackList = track_list_file.read().splitlines()
    return TrackList


# delete the temp files created by the interfaces to save the results in.
def remove_temp_files():
    files = glob.glob('TempFiles\\*')
    for f in files:
        os.remove(f)


try:
    # Run the GUIs to get what the user wants
    results = GUI.run_program()
    # remove the temp files creates by user interfaces
    remove_temp_files()

    if results[0] == 'exited':
        print('Exited the program.')
        sys.exit()
    else:
        print('loading the required model for classification of the new tweets.')
        # loading the desired model
        pipline = joblib.load('models\\' + str(results[0]) + '_' + str(results[1]) + '_' + str(results[2]) + '.sav')
        print('Done loading the model.')

        # connect the elasticsearch and get the instance that we will index the
        # new tweets based on it
        # shadi es = ConnectToES.connectToES()
        es = ConnectToES.connectToES()

        # open the browser on the kibana home page
        # shadi kibana_homepage = json.load(open('Objects/kibana_homepage_link.json'))
        kibana_homepage = json.load(open('Objects/kibana_homepage_link.json'))
        # shadi webbrowser.open(kibana_homepage)
        webbrowser.open(kibana_homepage)


except Exception as e:
    print('Exited the program.')
    sys.exit()


class StreamAPI(tweepy.StreamListener):
    # on_connect : Called once connected to streaming server.
    def on_connect(self):
        print('Connected to the Stream.')
        print('Getting Tweets')

    # When the stream receive a tweet object
    def on_status(self, status):
        # convert it to json
        json_data = status._json
        # we increase the total number of tweets by 1
        tweets[0] += 1
        # we won't pass all the information of the tweet to elasticsearch and kibana
        # we pass only the attributes that matters the most like the hashtags and user name etc ..

        # the user screen name with his real name. the problem is that the users don't put
        # there real name in the 'name' field so the names here is not accurate
        try:
            # if is is a retweet we won't take the tweet
            try:
                temp = json_data['retweeted_status']
            except:
                # if the extended_tweet has images or video we won't take the tweet
                try:
                    temp = json_data['extended_tweet']['entities']['media']
                except:
                    # if the normal tweet has images or video we won't take the tweet
                    try:
                        temp = json_data['entities']['media']
                    except:
                        # if the tweet isn't a reply to a user we take it
                        if json_data['in_reply_to_user_id'] is None:
                            # Quote Tweets are much like Retweets so we don't take them.
                            try:
                                temp = json_data['quoted_status']
                            except:
                                tweet = {'screen_name': json_data['user']['screen_name']}
                                # if the tweet contains an extended text (more than 140 chars)
                                Tweet_cleaned = ''
                                try:
                                    # we get the text of the tweet
                                    TweetText = json_data['extended_tweet']['full_text']
                                    # and the hashtags for the extended tweets
                                    if json_data['extended_tweet']['entities']:
                                        tweet['hashtags'] = json_data['extended_tweet']['entities']
                                    # remove the duplicated characters in the hashtags
                                    for i in range(0, len(tweet['hashtags'])):
                                        tweet['hashtags'][i]['text'] = re.sub(r'(.)\1+', r'\1\1',
                                                                              tweet['hashtags'][i]['text'])
                                        i += 1
                                except:
                                    # if the tweet is not extended then we will get the text and hashtags
                                    TweetText = json_data['text']
                                    # the hashtags the user used in his tweet
                                    if json_data['entities']['hashtags']:
                                        tweet['hashtags'] = json_data['entities']['hashtags']
                                        # remove the duplicated characters in the hashtags
                                        for i in range(0, len(tweet['hashtags'])):
                                            tweet['hashtags'][i]['text'] = re.sub(r'(.)\1+', r'\1\1',
                                                                                  tweet['hashtags'][i]['text'])
                                # we clean the tweet using the desired stemmer
                                Tweet_cleaned = CleanTweet.tweet_text_cleaner(TweetText, int(results[1]))
                                # if the tweet dose not has spams and contains more than one character after cleaning
                                if not (has_spam(TweetText)) and len(Tweet_cleaned) > 1:
                                    # we save the text into tweet object
                                    tweet['text'] = TweetText
                                    # prepare the tweet for prediction we have to put it in a list
                                    temp = list()
                                    temp.append(Tweet_cleaned)
                                    # let the model predict the tweet
                                    sentiment = pipline.predict(temp)
                                    # we add the sentiment to tweet object
                                    tweet['sentiment'] = sentiment[0]

                                    # adding the username
                                    tempName = json_data['user']['name']
                                    if tempName != "" and tempName != " ":
                                        tweet['name'] = tempName
                                    else:
                                        tweet['name'] = json_data['user']['screen_name']

                                    # the user location
                                    if json_data['place']:
                                        # we save the country code
                                        tweet['country_code'] = json_data['place']['country_code']
                                        # {SY, SA, KW, ...}
                                        tweet['country'] = json_data['place']['country']
                                        if json_data['place']['bounding_box']['coordinates']:
                                            # his exact location in that country (4 pairs of
                                            # latitude and longitude)

                                            tweet['coordinates'] = [
                                                (json_data['place']['bounding_box']['coordinates'][0][0][0] +
                                                 json_data['place']['bounding_box']['coordinates'][0][2][0]) / 2,
                                                (json_data['place']['bounding_box']['coordinates'][0][0][1] +
                                                 json_data['place']['bounding_box']['coordinates'][0][2][1]) / 2
                                            ]
                                    # we save the time when the tweet has been received
                                    if json_data['timestamp_ms']:
                                        tweet['timestamp_ms'] = json_data['timestamp_ms']

                                    # we increase the number of tweets added to elasticsearch
                                    tweets[1] += 1

                                    # we send the tweet to the index
                                    es.index(index="tweets_analyze", doc_type='_doc', body=tweet)
                                    # if we want to save tweet for manual classification
                                    # AddTweetToDataset(tweet['text'])

        except Exception as e:
            print(e)
            print('error in analysing the tweets .. retrying after 3 seconds ...')
            time.sleep(3)


# the default means the user wants the default tracklist to be used for streaming.
if results[3] == 'default':
    # we load the default track list
    trackList = GetTrackList()
    # we load the coordinates that include all the arab countries
    coordinates = json.load(open('Objects/coordinates_arab_countries.json'))
    while True:
        # get the api object
        api = TweetAuth.getTwitterAPI()
        # get the instance from the Stream_Api class
        streamAPI = StreamAPI()
        # creating the stream
        myStream = tweepy.Stream(auth=api.auth, listener=streamAPI)

        try:
            # filtering the stream for the key words, the language and location
            # Streams not terminate unless the connection is closed, blocking the thread.
            # the key words we want to search the twitter for, we can search up to 400 keyword

            myStream.filter(
                track=trackList,
                languages=['ar'],
                locations=coordinates,
                stall_warnings=True
            )
        # the errors which are triggered when there is a connection error
        # or the the connection to twitter has been broken which happens because
        # we are using the free license which means that we are restricted to
        # certain conditions.
        except Exception as e:
            print('\nError in Streaming ... !')
            print("total tweets Received = " + str(tweets[0]) + " and total tweets sent to Elasticsearch = " + str(
                tweets[1]))
            tweets = [0, 0]
            print('Restarting the stream ....')
            time.sleep(10)
            continue

# the user wants to stream based on his track list
elif len(results) > 3:
    # we get the tracklist which the user wrote in the interfaces
    user_tracklist = results[3:]
    while True:
        api = TweetAuth.getTwitterAPI()
        streamAPI = StreamAPI()
        myStream = tweepy.Stream(auth=api.auth, listener=streamAPI)
        # the filtering differ from the last filter cause we want a specific words
        # to appear in the tweets recived. so if we wrote the coordinates the tweets
        # received may be in the right coordinates but don't have the user track list words
        try:
            trackList = user_tracklist
            myStream.filter(
                track=trackList,
                languages=['ar'],
                stall_warnings=True
            )
        except:
            print('Error in streaming.')
            print("total tweets Received = " + str(tweets[0]) + " and total tweets sent to Elasticsearch = " + str(
                tweets[1]))
            print('Restarting the stream ....')
            tweets = [0, 0]
            time.sleep(10)
            continue
else:
    print('Exited the program.')
    sys.exit()

# Streams not terminate unless the connection is closed, blocking the thread.
