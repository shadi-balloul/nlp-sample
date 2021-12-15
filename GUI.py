from tkinter import *
import tkinter as tk
from tkinter.constants import *
import joblib
from PIL import Image
from PIL import ImageTk
import json

from FinalClassification import check_neutral_existence
from PreProcessing import remove_punctuations, remove_diacritics
from CleanTweet import tweet_text_cleaner

print("test....")

# read the names of databases, stemmers and classifiers that will be
# showed on the interfaces
gui_labels = json.load(open('Objects/interfaces_labels.json'))


# the main interface which let the user to select the wanted database, stemmer and classification
# algorithm before starting the streaming and has 2 options: the first is to Analysis tweets
# and the second is advanced streaming
def gui_for_database_stemmer_clfalgo():
    root = tk.Tk()
    HEIGHT = 700
    WIDTH = 1200
    # get screen width and height
    ws = root.winfo_screenwidth()  # width of the screen
    hs = root.winfo_screenheight()  # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws / 2) - (WIDTH / 2)
    y = (hs / 2) - (HEIGHT / 2)

    # set the dimensions of the screen
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (WIDTH, HEIGHT, x, y))
    root.title("Sentiment Analysis For Arabic Tweets")

    # the canvas is where we are going to place the main frame
    canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH, bg='#FFFFFF')
    canvas.pack()

    ##################################################

    # Functions

    ##################################################

    # return the selected buttons for the databases, stemmers and algorithms
    def database_selected():
        return database_number.get(), stemmer_number.get(), classalgo_number.get()

    # when the user clicks the normal streaming it will write on the temp file
    # the numbers for the database, stemmer and classifier with the word default
    # to know that the streaming will be default
    def streaming():
        # get the numbers selected
        db, stemmer, clf = database_selected()
        with open('TempFiles\\temp.txt', 'w+') as f:
            f.write(str(db) + '\n')
            f.write(str(stemmer) + '\n')
            f.write(str(clf) + '\n')
            f.write('default' + '\n')
        # close the window
        root.destroy()

    # when the user clicks the advanced streaming it will write on the temp file
    # the numbers for the database, stemmer and classifier with the word yes
    # to know that the streaming will be advanced
    def streaming_with_words():
        # get the numbers selected
        db, stemmer, clf = database_selected()
        with open('TempFiles\\temp.txt', 'w+') as f:
            f.write(str(db) + '\n')
            f.write(str(stemmer) + '\n')
            f.write(str(clf) + '\n')
            f.write('advanced' + '\n')
        # close the window
        root.destroy()

    # when the user clicks the analyze tweets it will write on the temp file
    # the numbers for the database, stemmer and classifier
    def sentiment_for_user_tweet():
        db, stemmer, clf = database_selected()
        with open('TempFiles\\temp.txt', 'w+') as f:
            f.write(str(db) + '\n')
            f.write(str(stemmer) + '\n')
            f.write(str(clf) + '\n')
        # button_start.configure(state=DISABLED)
        root.destroy()

    # when pressing the exit program we want to close the window
    def end_program():
        with open('TempFiles\\temp.txt', 'w+') as f:
            f.write('exited')
            root.destroy()

    # for pressing the X buttone on the top right button
    root.protocol('WM_DELETE_WINDOW', end_program)  # root is your root window

    # defining the numbers that will be changed when changing the selected button
    database_number = IntVar()
    stemmer_number = IntVar()
    classalgo_number = IntVar()

    ##################################################

    # adding logos

    ##################################################

    twitter_logo = Image.open("images/univ.png")
    twitter_logo = twitter_logo.resize((160, 160), Image.ANTIALIAS)
    twitter_logo = ImageTk.PhotoImage(twitter_logo)
    canvas.create_image(15, 0, image=twitter_logo, anchor=NW)

    univ_logo = Image.open("images/univ.png")
    univ_logo = univ_logo.resize((125, 125), Image.ANTIALIAS)
    univ_logo = ImageTk.PhotoImage(univ_logo)
    canvas.create_image(1150, 19, image=univ_logo, anchor=NE)

    ##################################################

    # the title

    ##################################################

    title_frame = tk.Frame(canvas, bg='white')
    title_frame.place(relx=0.25, rely=0.05, relwidth=0.5, relheight=0.35)

    title_label = tk.Label(title_frame, text='Sentiment Analysis for Arabic Tweets', font='new_roman 20 bold',
                           bg='lightblue', height=3)
    title_label.pack(fill='both')

    ##################################################

    # the options to pick before the stream

    ##################################################

    database_picker_frame = tk.Frame(canvas, bg='lightblue')
    database_picker_frame.place(relx=0.05, rely=0.25, relwidth=0.27, relheight=0.6)

    title_label = tk.Label(database_picker_frame, text='Select the desired database', bg='#6AABD2',
                           activebackground='lightblue',
                           font='new_roman 16')
    title_label.pack(fill='both')

    db_1 = Radiobutton(database_picker_frame, indicatoron=0, bg='#6AABD2', text=gui_labels['databases'][0],
                       variable=database_number, value=1,
                       command=database_selected, height=1, font=1, activebackground='#71c9f8')
    # db_1.place(relx=0.1, rely=0.1)

    db_2 = Radiobutton(database_picker_frame, indicatoron=0, text=gui_labels['databases'][1], variable=database_number,
                       value=2,
                       command=database_selected, height=1, font=1, bg='#6AABD2', activebackground='#71c9f8')
    db_2.place(relx=0.1, rely=0.23)

    db_3 = Radiobutton(database_picker_frame, indicatoron=0, text=gui_labels['databases'][2], variable=database_number,
                       value=3,
                       command=database_selected, height=1, font=1, bg='#6AABD2', activebackground='lightblue')
    db_3.place(relx=0.1, rely=0.36)

    db_4 = Radiobutton(database_picker_frame, indicatoron=0, text=gui_labels['databases'][3], variable=database_number,
                       value=4,
                       command=database_selected, height=1, font=1, bg='#6AABD2', activebackground='lightblue')
    db_4.place(relx=0.1, rely=0.49)

    db_5 = Radiobutton(database_picker_frame, indicatoron=0, text=gui_labels['databases'][4], variable=database_number,
                       value=5,
                       command=database_selected, height=1, font=1, bg='#6AABD2', activebackground='lightblue')
    db_5.place(relx=0.1, rely=0.62)

    db_6 = Radiobutton(database_picker_frame, indicatoron=0, text=gui_labels['databases'][5], variable=database_number,
                       value=6,
                       command=database_selected, height=1, font=1, bg='#6AABD2', activebackground='lightblue')
    db_6.place(relx=0.1, rely=0.75)

    db_all = Radiobutton(database_picker_frame, indicatoron=0, text=gui_labels['databases'][6],
                         variable=database_number, value=7,
                         command=database_selected, height=1, font=1, bg='#6AABD2', activebackground='lightblue')
    db_all.select()
    db_all.place(relx=0.1, rely=0.88)

    #####################################################################
    # stemmer picker
    #####################################################################

    stemmer_picker_frame = tk.Frame(canvas, bg='lightblue')
    stemmer_picker_frame.place(relx=0.39, rely=0.25, relwidth=0.25, relheight=0.6)

    title_label = tk.Label(stemmer_picker_frame, text='Select the desired stemmer', bg='#6AABD2',
                           font='new_roman 16')
    title_label.pack(fill='both')

    cog_stemmer = Radiobutton(stemmer_picker_frame, text=gui_labels['stemmers'][0], variable=stemmer_number, value=1,
                              command=database_selected,
                              activebackground='lightblue', indicatoron=0, bg='#6AABD2', font=1)
    cog_stemmer.place(relx=0.1, rely=0.1)

    snowball_stemmer = Radiobutton(stemmer_picker_frame, text=gui_labels['stemmers'][1], variable=stemmer_number,
                                   value=2,
                                   command=database_selected, font=1, indicatoron=0, bg='#6AABD2',
                                   activebackground='lightblue')

    snowball_stemmer.place(relx=0.1, rely=0.25)

    ISRI_stemmer = Radiobutton(stemmer_picker_frame, text=gui_labels['stemmers'][2], variable=stemmer_number, value=3,
                               command=database_selected, font=1, indicatoron=0, bg='#6AABD2',
                               activebackground='lightblue')
    ISRI_stemmer.place(relx=0.1, rely=0.4)

    improvedISRI = Radiobutton(stemmer_picker_frame, text=gui_labels['stemmers'][3], variable=stemmer_number, value=4,
                               command=database_selected, font=1, indicatoron=0, bg='#6AABD2',
                               activebackground='lightblue')
    improvedISRI.select()
    improvedISRI.place(relx=0.1, rely=0.55)

    khoja_stemmer = Radiobutton(stemmer_picker_frame, text=gui_labels['stemmers'][4], variable=stemmer_number, value=5,
                                command=database_selected, font=1, indicatoron=0, bg='#6AABD2',
                                activebackground='lightblue')

    khoja_stemmer.place(relx=0.1, rely=0.7)

    tashaphyne_stemmer = Radiobutton(stemmer_picker_frame, text=gui_labels['stemmers'][5], variable=stemmer_number,
                                     value=6,
                                     command=database_selected, font=1, indicatoron=0, bg='#6AABD2',
                                     activebackground='lightblue')

    tashaphyne_stemmer.place(relx=0.1, rely=0.85)

    ##################################################################
    # select the classification algorithm
    ##################################################################

    classificationalgo_picker_frame = tk.Frame(canvas, bg='lightblue')
    classificationalgo_picker_frame.place(relx=0.7, rely=0.25, relwidth=0.27, relheight=0.6)

    title_label = tk.Label(classificationalgo_picker_frame, text='Select the classification algorithm', bg='#6AABD2',
                           font='new_roman 14')
    title_label.pack(fill='both')

    linearSVC_algo = Radiobutton(classificationalgo_picker_frame, text=gui_labels['classifiers'][0],
                                 variable=classalgo_number, value=1,
                                 command=database_selected, activebackground='lightblue', indicatoron=0, bg='#6AABD2',
                                 font=1,
                                 highlightcolor='blue', borderwidth=1)

    linearSVC_algo.select()
    linearSVC_algo.place(relx=0.1, rely=0.1)

    SVC_algo = Radiobutton(classificationalgo_picker_frame, text=gui_labels['classifiers'][1],
                           variable=classalgo_number, value=2,
                           command=database_selected, font=1, indicatoron=0, bg='#6AABD2', activebackground='lightblue')

    SVC_algo.place(relx=0.1, rely=0.21)

    SGD_algo = Radiobutton(classificationalgo_picker_frame, text=gui_labels['classifiers'][2],
                           variable=classalgo_number, value=3,
                           command=database_selected, font=1, indicatoron=0, bg='#6AABD2', activebackground='lightblue')
    SGD_algo.place(relx=0.1, rely=0.32)

    MultinomialNB_algo = Radiobutton(classificationalgo_picker_frame, text=gui_labels['classifiers'][3],
                                     variable=classalgo_number,
                                     value=4,
                                     command=database_selected, font=1, indicatoron=0, bg='#6AABD2',
                                     activebackground='lightblue')
    MultinomialNB_algo.place(relx=0.1, rely=0.43)

    BernoulliNB_algo = Radiobutton(classificationalgo_picker_frame, text=gui_labels['classifiers'][4],
                                   variable=classalgo_number,
                                   value=5,
                                   command=database_selected, font=1, indicatoron=0, bg='#6AABD2',
                                   activebackground='lightblue')
    BernoulliNB_algo.place(relx=0.1, rely=0.54)

    DT_algo = Radiobutton(classificationalgo_picker_frame, text=gui_labels['classifiers'][5], variable=classalgo_number,
                          value=6,
                          command=database_selected, font=1, indicatoron=0, bg='#6AABD2', activebackground='lightblue')
    DT_algo.place(relx=0.1, rely=0.65)

    RT_algo = Radiobutton(classificationalgo_picker_frame, text=gui_labels['classifiers'][6], variable=classalgo_number,
                          value=7,
                          command=database_selected, font=1, indicatoron=0, bg='#6AABD2', activebackground='lightblue')

    RT_algo.place(relx=0.1, rely=0.76)

    KNN_algo = Radiobutton(classificationalgo_picker_frame, text=gui_labels['classifiers'][7],
                           variable=classalgo_number,
                           value=8,
                           command=database_selected, font=1, indicatoron=0, bg='#6AABD2', activebackground='lightblue')

    KNN_algo.place(relx=0.1, rely=0.87)

    ##################################################################
    # the buttons
    ##################################################################

    buttons_frame = tk.Frame(canvas, bg='white')
    buttons_frame.place(relx=0, rely=0.87, relwidth=1, relheight=0.8)

    button_start = Button(buttons_frame, text="Start Normal Streaming", width=22, height=3, bg='#89D997',
                          # activebackground='lightblue',
                          font='new_roman 11 bold', command=streaming)
    button_start.place(relx=0.11, rely=0)

    button_sa_tweet = Button(buttons_frame, text="Analyze Tweets", width=22, height=3, bg='#89D997',
                             # activebackground='lightblue',
                             font='new_roman 11 bold', command=sentiment_for_user_tweet)

    button_sa_tweet.place(relx=0.31, rely=0)

    button_stream_on_words = Button(buttons_frame, text="Start Advanced Streaming", width=22, height=3, bg='#89D997',
                                    # activebackground='lightblue',
                                    font='new_roman 11 bold', command=streaming_with_words)

    button_stream_on_words.place(relx=0.51, rely=0)

    button_quit = Button(buttons_frame, width=22, height=3, bg='lightpink', activebackground='pink',
                         text="Exit Program", command=end_program,
                         font='new_roman 11 bold')
    button_quit.place(relx=0.71, rely=0)

    root.mainloop()


# this interface is for the advanced streaming option, this interface will let the user
# stream on specified words of his choice
def gui_for_filteringwords():
    root = tk.Tk()
    HEIGHT = 700
    WIDTH = 800
    # get screen width and height
    ws = root.winfo_screenwidth()  # width of the screen
    hs = root.winfo_screenheight()  # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws / 2) - (WIDTH / 2)
    y = (hs / 2) - (HEIGHT / 2)

    # set the dimensions of the screen
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (WIDTH, HEIGHT, x, y))
    root.title("Sentiment Analysis For Arabic Tweets")

    canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH, bg='white')
    canvas.pack()

    ##################################################

    # Functions

    ##################################################

    # this function will check the words in the text field
    # and won't allow english words and numbers and for error will show
    # error label which include the error message
    def check_words(words):
        line_number = 0
        if textExample.compare("end-1c", "==", "1.0"):
            error_label.configure(text='Please Enter Some Arabic Words !',
                                  bg='lightpink')
            return False
        else:
            for word in words:
                line_number += 1
                if re.findall('[a-zA-Z]', word):
                    error_label.configure(
                        text='Error in line ' + str(line_number) + " words contain english characters!",
                        bg='lightpink')
                    return False
                elif re.findall('[0-9]', word):
                    error_label.configure(
                        text='Error in line ' + str(line_number) + " words contain english numbers!",
                        bg='lightpink')
                    return False
            error_label.configure(
                text='',
                bg='white')
            return True

    # this function that will triggered when pressing on start streaming
    # it will save the desired words in a temp1 file after checking them
    def start_advanced_streaming():
        result = textExample.get("1.0", "end")
        words = result.split()
        if check_words(words):
            words = list(dict.fromkeys(words))

            with open('TempFiles\\temp1.txt', 'w+', encoding='UTF-8') as f:
                for word in words:
                    word = remove_punctuations(word)
                    word = remove_diacritics(word)
                    f.write(word + '\n')
            root.destroy()

    # the cancel button will send you to the first form
    def return_to_first_form():
        with open('TempFiles\\temp1.txt', 'w+', encoding='UTF-8') as f:
            f.write(str('cancelled') + '\n')
        root.destroy()

    # the X button on the top right window will send you to the main interface
    # and close this interface
    root.protocol('WM_DELETE_WINDOW', return_to_first_form)  # root is your root window

    # the clear button will clear the words in the text field
    def clear_contents():
        error_label.configure(text='', bg='white')
        textExample.delete(1.0, END)

    ##################################################

    # adding logos

    ##################################################

    twitter_logo = Image.open("images\\logo.png")
    twitter_logo = twitter_logo.resize((160, 160), Image.ANTIALIAS)
    twitter_logo = ImageTk.PhotoImage(twitter_logo)
    canvas.create_image(15, 0, image=twitter_logo, anchor=NW)

    univ_logo = Image.open("images/univ.png")
    univ_logo = univ_logo.resize((125, 125), Image.ANTIALIAS)
    univ_logo = ImageTk.PhotoImage(univ_logo)
    canvas.create_image(750, 19, image=univ_logo, anchor=NE)

    ##################################################

    # the title

    ##################################################

    title_frame = tk.Frame(canvas, bg='white')
    title_frame.place(relx=0.24, rely=0.05, relwidth=0.5, relheight=0.38)

    title_label = tk.Label(title_frame, text='Sentiment Analysis for Arabic Tweets', font='new_roman 16 bold',
                           bg='lightblue', height=2)
    title_label.pack(fill='both')

    title_label = tk.Label(title_frame, text='Advanced Streaming', font='new_roman 16 bold',
                           bg='lightblue', height=2)
    title_label.pack(fill='both')

    ##################################################

    # the textbox

    ##################################################
    text_label_frame = tk.Frame(canvas, bg='white')
    text_label_frame.place(relx=0.05, rely=0.4, relwidth=0.4, relheight=0.4)

    text1_label = tk.Label(text_label_frame, text='Enter the words', font='new_roman 17 ',
                           bg='white', height=2)
    text1_label.pack(fill='both')

    text2_label = tk.Label(text_label_frame, text='of your choice', font='new_roman 17 ',
                           bg='white', height=2)
    text2_label.pack(fill='both')

    text3_label = tk.Label(text_label_frame, text='for the stream', font='new_roman 17',
                           bg='white', height=2)
    text3_label.pack(fill='both')

    error_label = Label(text_label_frame, text='', bg='white', font='new_roman 11',
                        height=2)
    error_label.pack(fill='x')

    text_frame = tk.Frame(canvas, bg='white')
    text_frame.place(relx=0.5, rely=0.3, relwidth=0.4, relheight=0.5)

    textExample = tk.Text(text_frame, height=18, bg='lightgrey', font='traditional_arabic 15', selectbackground='white',
                          wrap='word')
    textExample.pack()
    # change to ltr

    button_clear = Button(text_frame, text="Clear words", width=15, height=2, bg='lightblue',
                          activebackground='lightgreen',
                          font='new_roman 11 bold', command=clear_contents)
    button_clear.place(relx=0, rely=0.85, relwidth=1)

    # put the default text
    # textExample.redo('place your words here ...')

    ##################################################################
    # the buttons
    ##################################################################

    buttons_frame = tk.Frame(canvas, bg='white')
    buttons_frame.place(relx=0, rely=0.87, relwidth=1, relheight=0.8)

    button_start = Button(buttons_frame, text="Start Streaming", width=15, height=3, bg='#89D997',
                          activebackground='lightgreen',
                          font='new_roman 11 bold', command=start_advanced_streaming)
    button_start.place(relx=0.3, rely=0)

    button_quit = Button(buttons_frame, width=15, height=3, bg='lightpink', activebackground='pink', text="Cancel",
                         command=return_to_first_form,
                         font='new_roman 11 bold')
    button_quit.place(relx=0.5, rely=0)

    root.mainloop()
    return root


# this interface is for the second option which is analysing tweets, this interfaces
# with allow the users to enter a tweet and test it on the selected model
def gui_for_sa_tweet():
    root = tk.Tk()
    HEIGHT = 700
    WIDTH = 800
    # get screen width and height
    ws = root.winfo_screenwidth()  # width of the screen
    hs = root.winfo_screenheight()  # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws / 2) - (WIDTH / 2)
    y = (hs / 2) - (HEIGHT / 2)

    # set the dimensions of the screen
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (WIDTH, HEIGHT, x, y))
    root.title("Sentiment Analysis For Arabic Tweets")

    canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH, bg='white')
    canvas.pack()

    ##################################################

    # Functions

    ##################################################

    # clear the results labels for analysing new tweet
    def clear_results():
        tweet1_label.configure(text='', bg='white')
        tweet2_label.configure(text='', bg='white')
        results_label.configure(text='', bg='white')
        error_label.configure(text='', bg='white')

    # check the words entered which will not allow only english words or only
    # english numbers
    def check_words(words):
        if textExample.compare("end-1c", "==", "1.0"):
            clear_results()
            error_label.configure(text='Blank tweet, please enter a tweet text!',
                                  bg='lightpink')
            return False
        else:
            flag_words = False
            flag_numbers = False
            for word in words:
                if not re.findall('[a-zA-Z]', word):
                    flag_words = True
                    break
            for word in words:
                if not re.findall('[1-9]', word):
                    flag_numbers = True
                    break
            if not flag_words:
                clear_results()
                error_label.configure(
                    text='Please enter some Arabic words!',
                    bg='lightpink')
                return False
            elif not flag_numbers:
                clear_results()
                error_label.configure(
                    text='Please enter some Arabic words!',
                    bg='lightpink')
                return False
            else:
                error_label.configure(
                    text='',
                    bg='white')
                return True

    # train the model and make him ready for the classification
    with open('TempFiles\\temp.txt', 'r+', encoding='UTF-8') as f:
        content = f.readlines()
        options = [x.strip() for x in content]
    model_file = 'models\\' + str(options[0]) + '_' + str(options[1]) + '_' + str(options[2]) + '.sav'
    loaded_model = joblib.load(model_file)

    # get the sentiment for the tweet and print the results
    def sentiment_for_tweet():
        result = textExample.get("1.0", "end")
        words = result.split()
        if check_words(words):
            words = list(dict.fromkeys(words))
            tweet = ' '.join(words)
            tweet = tweet_text_cleaner(tweet, options[1])
            if tweet != "":
                result = loaded_model.predict([tweet])
                d = {'P': 'Positive', 'N': 'Negative', 'U': 'Neutral'}
                sa = d[result[0]]
                tweet1_label.configure(text='The Sentiment')
                tweet2_label.configure(text='Results')

                # check if the database contains the neutral class
                if check_neutral_existence(int(options[0])):

                    # for algos that don't probabilities
                    if int(options[2]) < 4:
                        prob = loaded_model.decision_function([tweet])
                        results_label.configure(text='the tweet is: \t' + sa + '\n\nThe scores are:\n\n'
                                                                               'Positive = ' + str(
                            round(prob[0][1], 3)) + '\n'
                                                    'Negative = ' + str(round(prob[0][0], 3)) + '\n'
                                                                                                'Neutral = ' + str(
                            round(prob[0][2], 3)),
                                                bg='lightblue')

                    else:
                        prob = loaded_model.predict_proba([tweet])
                        results_label.configure(text='the tweet is: \t' + sa + '\n\nThe probabilities are:\n\n'
                                                                               'Positive = ' + str(
                            round(prob[0][1], 3)) + '\n'
                                                    'Negative = ' + str(round(prob[0][0], 3)) + '\n'
                                                                                                'Neutral = ' + str(
                            round(prob[0][2], 3)),
                                                bg='lightblue')
                else:
                    # for algos that support probabilities
                    if int(options[2]) < 4:
                        prob = loaded_model.decision_function([tweet])[0]
                        if prob > 0:
                            neg_prob = prob
                            pos_prob = 1 - prob
                        else:
                            pos_prob = prob * (-1)
                            neg_prob = 1 - pos_prob

                        results_label.configure(text='the tweet is: \t' + sa + '\n\nThe scores are:\n\n'
                                                                               'Positive = ' + str(
                            round(pos_prob, 3)) + '\n'
                                                  'Negative = ' + str(round(neg_prob, 3)), bg='lightblue')

                    else:
                        prob = loaded_model.predict_proba([tweet])
                        prob = prob.tolist()[0]
                        results_label.configure(text='the tweet is: \t' + sa + '\n\nThe probabilities are:\n\n'
                                                                               'Positive = ' + str(
                            round(prob[1], 3)) + '\n'
                                                 'Negative = ' + str(round(prob[0], 3)),
                                                bg='lightblue')


            else:
                error_label.configure(
                    text='''Tweet's got no Sentiment''',
                    bg='lightpink')

    def return_to_first_form():
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', return_to_first_form)  # root is your root window

    def clear_contents():
        clear_results()
        textExample.delete(1.0, END)

    ##################################################

    # adding logos

    ##################################################

    twitter_logo = Image.open("images\\logo.png")
    twitter_logo = twitter_logo.resize((160, 160), Image.ANTIALIAS)
    twitter_logo = ImageTk.PhotoImage(twitter_logo)
    canvas.create_image(15, 0, image=twitter_logo, anchor=NW)

    hiast_logo = Image.open("images/univ.png")
    hiast_logo = hiast_logo.resize((125, 125), Image.ANTIALIAS)
    hiast_logo = ImageTk.PhotoImage(hiast_logo)
    canvas.create_image(750, 19, image=hiast_logo, anchor=NE)

    ##################################################

    # the title

    ##################################################

    title_frame = tk.Frame(canvas, bg='white')
    title_frame.place(relx=0.24, rely=0.05, relwidth=0.5, relheight=0.38)

    title_label = tk.Label(title_frame, text='Sentiment Analysis for Arabic Tweets', font='new_roman 16 bold',
                           bg='lightblue', height=2)
    title_label.pack(fill='both')

    title_label = tk.Label(title_frame, text='Tweets Tester', font='new_roman 16 bold',
                           bg='lightblue', height=2)
    title_label.pack(fill='both')

    ##################################################

    # the textbox

    ##################################################
    tweet_label_frame = tk.Frame(canvas, bg='white')
    tweet_label_frame.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.3)

    text1_label = tk.Label(tweet_label_frame, text='Enter the tweet', font='new_roman 17',
                           bg='white', height=2)
    text1_label.place(relx=0, rely=0.4, relwidth=0.3, relheight=0.2)

    error_label = Label(tweet_label_frame, text='', bg='white', font='new_roman 12',
                        height=2)
    error_label.place(relx=0.3, rely=0.66, relwidth=0.56, relheight=0.2)

    text_frame = tk.Frame(tweet_label_frame, bg='white')
    text_frame.place(relx=0.3, rely=0.3, relwidth=0.7, relheight=0.4)

    textExample = tk.Text(text_frame, height=18, bg='lightgrey', font='traditional_arabic 15', wrap='word')
    textExample.place(relx=0, relwidth=0.8)

    button_clear_tweet = Button(text_frame, text="Clear", width=15, height=3, bg='lightblue',
                                activebackground='lightblue',
                                font='new_roman 11 bold', command=clear_contents)
    button_clear_tweet.place(relx=0.8, rely=0.25, relwidth=0.2, relheight=0.5)

    # change to ltr

    ##################################################################
    # the results frame
    ##################################################################

    tweet_result_frame = tk.Frame(canvas, bg='white')
    tweet_result_frame.place(relx=0.01, rely=0.45, relwidth=0.95, relheight=0.3)

    tweet1_label = tk.Label(tweet_result_frame, text='', font='new_roman 16',
                            bg='white', height=2)
    tweet1_label.place(relx=0.05, rely=0.35, relwidth=0.25, relheight=0.2)

    tweet2_label = tk.Label(tweet_result_frame, text='', font='new_roman 16',
                            bg='white', height=2)
    tweet2_label.place(relx=0.05, rely=0.55, relwidth=0.25, relheight=0.2)

    results_label = tk.Label(tweet_result_frame, text='', font='new_roman 16',
                             bg='white', height=2)
    results_label.place(relx=0.4, rely=0.05, relwidth=0.45, relheight=0.9)

    ##################################################################
    # the buttons
    ##################################################################

    buttons_frame = tk.Frame(canvas, bg='white')
    buttons_frame.place(relx=0, rely=0.87, relwidth=1, relheight=0.8)

    button_start = Button(buttons_frame, text="Get the Sentiment", width=15, height=3, bg='#89D997',
                          activebackground='lightgreen',
                          font='new_roman 11 bold', command=sentiment_for_tweet)
    button_start.place(relx=0.3, rely=0)

    button_quit = Button(buttons_frame, width=15, height=3, bg='lightpink', activebackground='pink', text="Cancel",
                         command=return_to_first_form,
                         font='new_roman 11 bold')
    button_quit.place(relx=0.5, rely=0)

    root.mainloop()
    return root


# the main function the run all the interfaces
def run_program():
    result = []
    # run the main interface and put the results in temp.txt
    gui_for_database_stemmer_clfalgo()
    # open the temp.txt to know what the user requested
    with open('TempFiles\\temp.txt', 'r+', encoding='UTF-8') as f:
        content = f.readlines()
        options = [x.strip() for x in content]
        # if the user wants to exit the program
        if options[0] == 'exited':
            result.append('exited')
            return result
    # if the user wants to analyse Tweets
    if len(options) == 3:
        # run the analyzing tweets interface
        gui_for_sa_tweet()
        # stay in the program until the user exit the analyzing interface
        return run_program()
    # if the use wants to do advanced streaming
    elif options[3] == 'advanced':
        # run the advanced streaming interface and save the results in temp1.txt
        gui_for_filteringwords()
        with open('TempFiles\\temp1.txt', 'r+', encoding='UTF-8') as f:
            content = f.readlines()
            track_list = [x.strip() for x in content]
            # if the user entered the track list words he wants
            if track_list[0] != 'cancelled':
                result.append(options[0])
                result.append(options[1])
                result.append(options[2])
                result = result + track_list
                return result
            # if the user cancelled the operation
            else:
                # stay in the program until the user exit the analyzing interface
                return run_program()
    # if the use wants to do normal streaming
    else:
        result.append(options[0])
        result.append(options[1])
        result.append(options[2])
        result.append('default')
        return result

