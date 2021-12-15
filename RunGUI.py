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
        # pipline = joblib.load('models\\' + str(results[0]) + '_' + str(results[1]) + '_' + str(results[2]) + '.sav')
        print('Done loading the model.')

        # connect the elasticsearch and get the instance that we will index the
        # new tweets based on it
        # es = ConnectToES.connectToES()

        # open the browser on the kibana home page
        # kibana_homepage = json.load(open('Objects/kibana_homepage_link.json'))
        # webbrowser.open(kibana_homepage)


except Exception as e:
    print('Exited the program.')
    sys.exit()

