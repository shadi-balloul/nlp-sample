import glob
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

# we ignore the warnings which will be triggered like the neutral class has 0 precision
# when we test the classifiers
warnings.filterwarnings('ignore')


# check if the database has neutral class
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


# get the best k for the algorithm knn and get the best max depth for the
# decision trees and random forest trees
def best_k_max_depth(algo, x_train, y_train, x_test, y_test, dataset_number):
    best_depth_accuracy = 0
    best_depth = 0
    if algo == 'DT':
        # for all the values between one and 200 we are creating a 200 classifiers
        # and getting the best accuracy then for the best accurcy
        # we will return the max_depth selected
        for k1 in range(1, 200):
            pipeline = Pipeline([
                ('main_vect', TfidfVectorizer(
                    analyzer='word', lowercase=False,
                    ngram_range=(1, 2)
                )),
                ('main_clf', DecisionTreeClassifier(max_depth=k1)),
            ])
            pipeline.fit(x_train, y_train)
            y_predicted = pipeline.predict(x_test)
            if not check_neutral_existence(dataset_number):
                cr = metrics.classification_report(y_test, y_predicted, target_names=['P', 'N'], output_dict=True)
            else:
                cr = metrics.classification_report(y_test, y_predicted, target_names=['P', 'N', 'U'], output_dict=True)
            # get the accuracy
            clf_results = cr['accuracy']
            # if the best depth accuracy is lower than this k accuracy
            if best_depth_accuracy < clf_results:
                # we save the new accuracy and max_depth
                best_depth_accuracy = clf_results
                best_depth = k1
        return best_depth

    elif algo == "KNN":
        best_k_accuracy = 0
        best_k = 1
        for k2 in range(1, 200):
            pipeline = Pipeline([
                ('main_vect', TfidfVectorizer(
                    analyzer='word', lowercase=False,
                    ngram_range=(1, 2)

                )),
                ('main_clf', KNeighborsClassifier(k2)),
            ])
            pipeline.fit(x_train, y_train)
            y_predicted = pipeline.predict(x_test)
            if not check_neutral_existence(dataset_number):
                cr = metrics.classification_report(y_test, y_predicted, target_names=['P', 'N'], output_dict=True)
            else:
                cr = metrics.classification_report(y_test, y_predicted, target_names=['P', 'N', 'U'], output_dict=True)
            clf_results = cr['accuracy']
            if best_k_accuracy < clf_results:
                best_k_accuracy = clf_results
                best_k = k2
        return best_k
    else:
        best_depth_accuracy = 0
        best_depth = 1
        for k3 in range(1, 200):
            pipeline = Pipeline([
                ('main_vect', TfidfVectorizer(
                    analyzer='word', lowercase=False,
                    ngram_range=(1, 2)

                )),
                ('main_clf', RandomForestClassifier(max_depth=k3)),
            ])
            pipeline.fit(x_train, y_train)
            y_predicted = pipeline.predict(x_test)
            if not check_neutral_existence(dataset_number):
                cr = metrics.classification_report(y_test, y_predicted, target_names=['P', 'N'], output_dict=True)
            else:
                cr = metrics.classification_report(y_test, y_predicted, target_names=['P', 'N', 'U'], output_dict=True)
            clf_results = cr['accuracy']
            if best_depth_accuracy < clf_results:
                best_depth_accuracy = clf_results
                best_depth = k3
        return best_depth


# this function returns the classifier and for the knn, decision tree and random trees we
# return classifiers with the best k and max depth for them
def classifier_based_on_number(clf_number, x_train, y_train, x_test, y_test, dataset_number):
    classifiers = [LinearSVC(), SVC(), SGDClassifier(), MultinomialNB(), BernoulliNB()]
    if clf_number <= 5:
        return classifiers[clf_number - 1]
    else:
        if clf_number == 6:
            best_depth = best_k_max_depth('DT', x_train, y_train, x_test, y_test, dataset_number)
            return DecisionTreeClassifier(max_depth=best_depth)
        elif clf_number == 7:
            best_depth = best_k_max_depth('RT', x_train, y_train, x_test, y_test, dataset_number)
            return RandomForestClassifier(max_depth=best_depth)
        else:
            best_k = best_k_max_depth('KNN', x_train, y_train, x_test, y_test, dataset_number)
            return KNeighborsClassifier(n_neighbors=best_k)


# loading the data in parallel to lower the time needed for training
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
            if not check_neutral_existence:
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
        multiprocessing.freeze_support()
        return output
    except Exception as e:
        print(e)
        return -1


# this function will do the classification and returns the pipeline
# to be saved in the trained models
def SA(dataset_number, stemmer_number, classifier_number):
    # get the preprocessed data
    output = load_data(dataset_number, stemmer_number)
    try:
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
            if check_neutral_existence:
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
        # get the classifier
        clf = classifier_based_on_number(classifier_number, x_train, y_train, x_test, y_test, dataset_number)
        pipeline = Pipeline([
            ('main_vect', TfidfVectorizer(
                analyzer='word', lowercase=False,
                ngram_range=(1, 2)
            )),
            ('main_clf', clf),
        ])
        # train the model
        pipeline.fit(x_train, y_train)
        # return the pipeline
        return pipeline
    except Exception as e:
        print(e)


# this function will save the models for each classifier
def save_models(dataset_number, stemmer_number):
    # get the preprocessed data and merge the data into training and testing data
    output = load_data(dataset_number, stemmer_number)
    try:
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
            if check_neutral_existence:
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
        # for each classifier we are training the model and saving it in the models directory
        for i in range(1, 9):
            clf = classifier_based_on_number(i, x_train, y_train, x_test, y_test, dataset_number)
            pipeline = Pipeline([
                ('main_vect', TfidfVectorizer(
                    analyzer='word', lowercase=False,
                    ngram_range=(1, 2)
                )),
                ('main_clf', clf),
            ])
            pipeline.fit(x_train, y_train)
            # saving the model in as databaseNumber_stemmerNumber_ClassificationAlgorithmNumber.sav
            joblib.dump(pipeline, 'models\\' + str(dataset_number) + '_' + str(stemmer_number) + '_' + str(i) + '.sav')

    except Exception as e:
        print(e)


# we run this for saving all the models
if __name__ == '__main__':
    for database in range(1, 8):
        for stemmer in range(1, 7):
            save_models(database, stemmer)
