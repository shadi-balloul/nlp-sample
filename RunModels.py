import joblib
from CleanTweet import tweet_text_cleaner
import glob
import json
import multiprocessing
import joblib
from pexecute.process import ProcessLoom
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from FeaturesExtraction import read_csv
import warnings
from datetime import datetime

"""
# model_file = 'models\\' + str(options[0]) + '_' + str(options[1]) + '_' + str(options[2]) + '.sav'
result = "الحياة جميلة"
words = result.split()
words = list(dict.fromkeys(words))
tweet = ' '.join(words)
tweet = tweet_text_cleaner(tweet, 6)
print(tweet)
model_file1 = 'models\\7_3_1.sav'
loaded_model1 = joblib.load(model_file1)
print(loaded_model1.get_params)
print(type(loaded_model1['main_clf']))
lscv = loaded_model1['main_clf']
# df = lscv.score()
loaded_model1.predict([tweet])

if tweet != "":
    result = loaded_model1.predict([tweet])
    d = {'P': 'Positive', 'N': 'Negative', 'U': 'Neutral'}
    sa = d[result[0]]
prob = loaded_model1.decision_function([tweet])
print('sa= ' + sa)
print('positive ' + str(round(prob[0][0], 3)))
print('negative ' + str(round(prob[0][1], 3)))
print('neutral ' + str(round(prob[0][2], 3)))

"""

def check_neutral_existence(dataset_number):
    if dataset_number == 7:
        return True
    else:
        # get the files names in the data file
        list_of_files = glob.glob('Data/*')
        results = dict()
        # for each file
        for i in range(0, len(list_of_files)):
            file = list_of_files[i]
            # splitting to get the file name without the path
            temp = file.split('\\')[1]
            # getting the number of the database for the file
            number = int(temp.split('_')[0])
            # adding the number to the dictionary
            # and for the databases that have been added we are increasing
            # the values by one to know exactly how many file for each database
            results[number] = results.get(number, 0) + 1
        # if the selected data has 6 files is will contain neutral class
        # cause if will have 2 files for each class
        if results[dataset_number] == 6:
            return True
        else:
            return False



def load_data(dataset, stemmer):
    try:
        # for the database 7 which is the merging of all below databases
        if dataset == 7:
            loom = ProcessLoom(max_runner_cap=12)
            work = \
                [
                    (read_csv, ('Data/1_train_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/1_train_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/1_train_neu.csv', 'U', stemmer)),
                    (read_csv, ('Data/1_test_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/1_test_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/1_test_neu.csv', 'U', stemmer)),

                    (read_csv, ('Data/2_train_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/2_train_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/2_train_neu.csv', 'U', stemmer)),
                    (read_csv, ('Data/2_test_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/2_test_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/2_test_neu.csv', 'U', stemmer)),

                    (read_csv, ('Data/3_train_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/3_train_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/3_train_neu.csv', 'U', stemmer)),
                    (read_csv, ('Data/3_test_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/3_test_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/3_test_neu.csv', 'U', stemmer)),

                    (read_csv, ('Data/4_train_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/4_train_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/4_test_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/4_test_neg.csv', 'N', stemmer)),

                    (read_csv, ('Data/5_train_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/5_train_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/5_train_neu.csv', 'U', stemmer)),
                    (read_csv, ('Data/5_test_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/5_test_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/5_test_neu.csv', 'U', stemmer)),

                    (read_csv, ('Data/6_train_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/6_train_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/6_train_neu.csv', 'U', stemmer)),
                    (read_csv, ('Data/6_test_pos.csv', 'P', stemmer)),
                    (read_csv, ('Data/6_test_neg.csv', 'N', stemmer)),
                    (read_csv, ('Data/6_test_neu.csv', 'U', stemmer)),

                ]
        else:
            # for the databases that don't have the neutral class
            if not check_neutral_existence(dataset):
                loom = ProcessLoom(max_runner_cap=4)
                train_pos = 'Data/' + str(dataset) + '_train_pos.csv'
                train_neg = 'Data/' + str(dataset) + '_train_neg.csv'
                test_pos = 'Data/' + str(dataset) + '_test_pos.csv'
                test_neg = 'Data/' + str(dataset) + '_test_neg.csv'

                work = \
                    [
                        (read_csv, (train_pos, 'P', stemmer)),
                        (read_csv, (train_neg, 'N', stemmer)),
                        (read_csv, (test_pos, 'P', stemmer)),
                        (read_csv, (test_neg, 'N', stemmer)),
                    ]
            # for the databases that have all three classes
            else:
                loom = ProcessLoom(max_runner_cap=6)
                train_pos = 'Data/' + str(dataset) + '_train_pos.csv'
                train_neg = 'Data/' + str(dataset) + '_train_neg.csv'
                train_neu = 'Data/' + str(dataset) + '_train_neu.csv'
                test_pos = 'Data/' + str(dataset) + '_test_pos.csv'
                test_neg = 'Data/' + str(dataset) + '_test_neg.csv'
                test_neu = 'Data/' + str(dataset) + '_test_neu.csv'

                work = \
                    [
                        (read_csv, (train_pos, 'P', stemmer)),
                        (read_csv, (train_neg, 'N', stemmer)),
                        (read_csv, (train_neu, 'U', stemmer)),
                        (read_csv, (test_pos, 'P', stemmer)),
                        (read_csv, (test_neg, 'N', stemmer)),
                        (read_csv, (test_neu, 'U', stemmer)),
                    ]

        loom.add_work(work)
        output = loom.execute()
        #multiprocessing.freeze_support()
        return output
    except Exception as e:
        print(e)
        return -1


def TSA_Aquarcy(dataset_number, stemmer_number, loaded_model):
    output = load_data(dataset_number, stemmer_number)
    # get the output from applying pre processing for the data
    # and save them into x_train x_test and y_train y_test
    check_neutral_existenc = check_neutral_existence(dataset_number)

    if dataset_number == 7:
        pos_train_data_1, pos_train_labels_1 = output[0]['output']
        neg_train_data_1, neg_train_labels_1 = output[1]['output']
        neu_train_data_1, neu_train_labels_1 = output[2]['output']
        pos_test_data_1, pos_test_labels_1 = output[3]['output']
        neg_test_data_1, neg_test_labels_1 = output[4]['output']
        neu_test_data_1, neu_test_labels_1 = output[5]['output']

        pos_train_data_2, pos_train_labels_2 = output[6]['output']
        neg_train_data_2, neg_train_labels_2 = output[7]['output']
        neu_train_data_2, neu_train_labels_2 = output[8]['output']
        pos_test_data_2, pos_test_labels_2 = output[9]['output']
        neg_test_data_2, neg_test_labels_2 = output[10]['output']
        neu_test_data_2, neu_test_labels_2 = output[11]['output']

        pos_train_data_3, pos_train_labels_3 = output[12]['output']
        neg_train_data_3, neg_train_labels_3 = output[13]['output']
        neu_train_data_3, neu_train_labels_3 = output[14]['output']
        pos_test_data_3, pos_test_labels_3 = output[15]['output']
        neg_test_data_3, neg_test_labels_3 = output[16]['output']
        neu_test_data_3, neu_test_labels_3 = output[17]['output']

        pos_train_data_4, pos_train_labels_4 = output[18]['output']
        neg_train_data_4, neg_train_labels_4 = output[19]['output']
        pos_test_data_4, pos_test_labels_4 = output[20]['output']
        neg_test_data_4, neg_test_labels_4 = output[21]['output']

        pos_train_data_5, pos_train_labels_5 = output[22]['output']
        neg_train_data_5, neg_train_labels_5 = output[23]['output']
        neu_train_data_5, neu_train_labels_5 = output[24]['output']
        pos_test_data_5, pos_test_labels_5 = output[25]['output']
        neg_test_data_5, neg_test_labels_5 = output[26]['output']
        neu_test_data_5, neu_test_labels_5 = output[27]['output']

        pos_train_data_6, pos_train_labels_6 = output[28]['output']
        neg_train_data_6, neg_train_labels_6 = output[29]['output']
        neu_train_data_6, neu_train_labels_6 = output[30]['output']
        pos_test_data_6, pos_test_labels_6 = output[31]['output']
        neg_test_data_6, neg_test_labels_6 = output[32]['output']
        neu_test_data_6, neu_test_labels_6 = output[33]['output']

        pos_train = pos_train_data_1 + pos_train_data_2 + pos_train_data_3 + pos_train_data_5 + pos_train_data_6 + pos_train_data_4
        neg_train = neg_train_data_1 + neg_train_data_2 + neg_train_data_3 + neg_train_data_5 + neg_train_data_6 + neg_train_data_4
        neu_train = neu_train_data_1 + neu_train_data_2 + neu_train_data_3 + neu_train_data_5 + neu_train_data_6

        pos_train_labels = pos_train_labels_1 + pos_train_labels_2 + pos_train_labels_3 + pos_train_labels_5 + pos_train_labels_6 + pos_train_labels_4
        neg_train_labels = neg_train_labels_1 + neg_train_labels_2 + neg_train_labels_3 + neg_train_labels_5 + neg_train_labels_6 + neg_train_labels_4
        neu_train_labels = neu_train_labels_1 + neu_train_labels_2 + neu_train_labels_3 + neu_train_labels_5 + neu_train_labels_6

        pos_test = pos_test_data_1 + pos_test_data_2 + pos_test_data_3 + pos_test_data_5 + pos_test_data_6 + pos_test_data_4
        neg_test = neg_test_data_1 + neg_test_data_2 + neg_test_data_3 + neg_test_data_5 + neg_test_data_6 + neg_test_data_4
        neu_test = neu_test_data_1 + neu_test_data_2 + neu_test_data_3 + neu_test_data_5 + neu_test_data_6

        pos_test_labels = pos_test_labels_1 + pos_test_labels_2 + pos_test_labels_3 + pos_test_labels_5 + pos_test_labels_6 + pos_test_labels_4
        neg_test_labels = neg_test_labels_1 + neg_test_labels_2 + neg_test_labels_3 + neg_test_labels_5 + neg_test_labels_6 + neg_test_labels_4
        neu_test_labels = neu_test_labels_1 + neu_test_labels_2 + neu_test_labels_3 + neu_test_labels_5 + neu_test_labels_6

        x_train = pos_train + neg_train + neu_train

        y_train = pos_train_labels + neg_train_labels + neu_train_labels

        x_test = pos_test + neg_test + neu_test

        y_test = pos_test_labels + neg_test_labels + neu_test_labels
    else:
        if check_neutral_existenc:
            pos_train_data, pos_train_labels = output[0]['output']
            neg_train_data, neg_train_labels = output[1]['output']
            neu_train_data, neu_train_labels = output[2]['output']
            pos_test_data, pos_test_labels = output[3]['output']
            neg_test_data, neg_test_labels = output[4]['output']
            neu_test_data, neu_test_labels = output[5]['output']

            x_train = pos_train_data + neg_train_data + neu_train_data

            y_train = pos_train_labels + neg_train_labels + neu_train_labels

            x_test = pos_test_data + neg_test_data + neu_test_data

            y_test = pos_test_labels + neg_test_labels + neu_test_labels

        else:
            pos_train_data, pos_train_labels = output[0]['output']
            neg_train_data, neg_train_labels = output[1]['output']
            pos_test_data, pos_test_labels = output[2]['output']
            neg_test_data, neg_test_labels = output[3]['output']

            x_train = pos_train_data + neg_train_data
            y_train = pos_train_labels + neg_train_labels

            x_test = pos_test_data + neg_test_data

            y_test = pos_test_labels + neg_test_labels

    # printing data info
    print('train data size:{}\ttest data size:{}'.format(len(y_train), len(y_test)))
    print('train data: # of pos:{}\t# of neg:{}\t'.format(y_train.count('P'), y_train.count('N')))
    print('test data: # of pos:{}\t# of neg:{}\t'.format(y_test.count('P'), y_test.count('N')))
    print('------------------------------------')

    # get the classifier with the best parameter for the knn, DT, RFT classifiers
    # clf = classifier_based_on_number(classifier_number, x_train, y_train, x_test, y_test, dataset_number)

    # load the classifier in the pipeline
    """
    pipeline = Pipeline([
        ('main_vect', TfidfVectorizer(
            analyzer='word', lowercase=False,
            ngram_range=(1, 2)
        )),
        ('main_clf', clf),
    ])
   

    # train the model which train it after applying TF-IDF on it using the pipeline automatically
    pipeline.fit(x_train, y_train)

    # get the features
    feature_names = pipeline.named_steps['main_vect'].get_feature_names()

    # print the number of features extracted
    print('features:', )
    print(len(feature_names), 'are kept')
    print('features are selected')
     """
    # test the model which test it after applying TF-IDF on it using the pipeline automatically
    #y_predicted = pipeline.predict(x_test)
    y_predicted = loaded_model.predict(x_test)

    # printing the classification report for the classifier
    if check_neutral_existenc:
        # Print the classification report
        cr = metrics.classification_report(y_test, y_predicted,
                                            target_names=['P', 'N', 'U'], output_dict=True)
        # print(cr)
    else:
        # Print the classification report
        cr = metrics.classification_report(y_test, y_predicted,
                                            target_names=['P', 'N'],  output_dict=True)
        # print(cr)

    # Print the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    # print(cm)
    return cr, cm
    #print('# of features:', len(feature_names))

    # printing first 100 tweet for observation




if __name__ == '__main__':







    for database in range(7, 8):
        for stemmer in range(6, 7):
            for clf in range(4, 9):
                try:
                    # datetime object containing current date and time
                    now = datetime.now()

                    print("now =", now)

                    # dd/mm/YY H:M:S
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    print("date and time =", dt_string)

                    #if database != 4:
                    if stemmer != 5:
                            f = 'models\\' + str(database) + '_' + str(stemmer) + '_' + str(clf) +'.sav'
                            print("f")
                            print(f)
                            lm = joblib.load(f)
                            cr, cm = TSA_Aquarcy(database, stemmer, lm)

                            res = 'results/cr-' +  str(database) + '_' + str(stemmer) + '_' + str(clf) +'.json'
                            print("res")
                            print(res)
                            with open(res, "w", encoding="utf8") as outfile:
                                json.dump(cr, outfile)

                    # datetime object containing current date and time
                    now = datetime.now()

                    print("now =", now)

                    # dd/mm/YY H:M:S
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    print("date and time =", dt_string)


                except Exception as e:
                  print(e)
                  print('Error: ')
