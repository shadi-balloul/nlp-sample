import collections
import glob
import multiprocessing
import sys
from timeit import default_timer as timer
import nltk
from nltk import SklearnClassifier
from nltk.metrics.scores import f_measure, precision, recall
from pexecute.process import ProcessLoom
from FeaturesExtraction import *
from FinalClassification import classifier_based_on_number, check_neutral_existence


# preprocessing the data for each file in the dataset
# after we separated it into training and testing for each class
def pre_processing_data_parallel(dataset, stemmer, ngrams):
    # loom:This takes a number of tasks and executes them using a pool of threads/process. max_runner_cap: is the
    # number of maximum threads/processes to run at a time. You can add as many as functions you want, but only n
    # functions will run at a time in parallel, n is the max_runner_cap
    try:
        if dataset == 7:
            # for the dataset 7 which is the including for all the 1-6 datasets
            loom = ProcessLoom(max_runner_cap=12)
            work = \
                [
                    (load_data_and_features, ('Data/1_train_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/1_train_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/1_train_neu.csv', 'U', stemmer, ngrams)),
                    (load_data_and_features, ('Data/1_test_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/1_test_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/1_test_neu.csv', 'U', stemmer, ngrams)),

                    (load_data_and_features, ('Data/2_train_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/2_train_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/2_train_neu.csv', 'U', stemmer, ngrams)),
                    (load_data_and_features, ('Data/2_test_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/2_test_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/2_test_neu.csv', 'U', stemmer, ngrams)),

                    (load_data_and_features, ('Data/3_train_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/3_train_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/3_train_neu.csv', 'U', stemmer, ngrams)),
                    (load_data_and_features, ('Data/3_test_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/3_test_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/3_test_neu.csv', 'U', stemmer, ngrams)),

                    (load_data_and_features, ('Data/4_train_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/4_train_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/4_test_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/4_test_neg.csv', 'N', stemmer, ngrams)),

                    (load_data_and_features, ('Data/5_train_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/5_train_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/5_train_neu.csv', 'U', stemmer, ngrams)),
                    (load_data_and_features, ('Data/5_test_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/5_test_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/5_test_neu.csv', 'U', stemmer, ngrams)),

                    (load_data_and_features, ('Data/6_train_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/6_train_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/6_train_neu.csv', 'U', stemmer, ngrams)),
                    (load_data_and_features, ('Data/6_test_pos.csv', 'P', stemmer, ngrams)),
                    (load_data_and_features, ('Data/6_test_neg.csv', 'N', stemmer, ngrams)),
                    (load_data_and_features, ('Data/6_test_neu.csv', 'U', stemmer, ngrams)),

                ]
        else:
            # if the dataset contains only positive and negative classes
            if not check_neutral_existence:
                loom = ProcessLoom(max_runner_cap=4)
                train_pos = 'Data/' + str(dataset) + '_train_pos.csv'
                train_neg = 'Data/' + str(dataset) + '_train_neg.csv'
                test_pos = 'Data/' + str(dataset) + '_test_pos.csv'
                test_neg = 'Data/' + str(dataset) + '_test_neg.csv'

                work = \
                    [
                        (load_data_and_features, (train_pos, 'P', stemmer, ngrams)),
                        (load_data_and_features, (train_neg, 'N', stemmer, ngrams)),
                        (load_data_and_features, (test_pos, 'P', stemmer, ngrams)),
                        (load_data_and_features, (test_neg, 'N', stemmer, ngrams)),
                    ]
            # if the data set include 3 classes
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
                        (load_data_and_features, (train_pos, 'P', stemmer, ngrams)),
                        (load_data_and_features, (train_neg, 'N', stemmer, ngrams)),
                        (load_data_and_features, (train_neu, 'U', stemmer, ngrams)),
                        (load_data_and_features, (test_pos, 'P', stemmer, ngrams)),
                        (load_data_and_features, (test_neg, 'N', stemmer, ngrams)),
                        (load_data_and_features, (test_neu, 'U', stemmer, ngrams)),
                    ]

        loom.add_work(work)
        output = loom.execute()
        multiprocessing.freeze_support()
        return output
    except Exception as e:
        print(e)
        return -1



# do the classification with printing the results on a file
def SA_using_ngrams(dataset_number, stemmer, classifier_number, n):
    output = pre_processing_data_parallel(dataset_number, stemmer, n)
    # get the output from applying pre processing for the data
    # and save them into x_train x_test and y_train y_test

    if dataset_number == 7:
        pos_train_data_1, pos_train_features_1 = output[0]['output']
        neg_train_data_1, neg_train_features_1 = output[1]['output']
        neu_train_data_1, neu_train_features_1 = output[2]['output']
        pos_test_data_1, pos_test_features_1 = output[3]['output']
        neg_test_data_1, neg_test_features_1 = output[4]['output']
        neu_test_data_1, neu_test_features_1 = output[5]['output']

        pos_train_data_2, pos_train_features_2 = output[6]['output']
        neg_train_data_2, neg_train_features_2 = output[7]['output']
        neu_train_data_2, neu_train_features_2 = output[8]['output']
        pos_test_data_2, pos_test_features_2 = output[9]['output']
        neg_test_data_2, neg_test_features_2 = output[10]['output']
        neu_test_data_2, neu_test_features_2 = output[11]['output']

        pos_train_data_3, pos_train_features_3 = output[12]['output']
        neg_train_data_3, neg_train_features_3 = output[13]['output']
        neu_train_data_3, neu_train_features_3 = output[14]['output']
        pos_test_data_3, pos_test_features_3 = output[15]['output']
        neg_test_data_3, neg_test_features_3 = output[16]['output']
        neu_test_data_3, neu_test_features_3 = output[17]['output']

        pos_train_data_4, pos_train_features_4 = output[18]['output']
        neg_train_data_4, neg_train_features_4 = output[19]['output']
        pos_test_data_4, pos_test_features_4 = output[20]['output']
        neg_test_data_4, neg_test_features_4 = output[21]['output']

        pos_train_data_5, pos_train_features_5 = output[22]['output']
        neg_train_data_5, neg_train_features_5 = output[23]['output']
        neu_train_data_5, neu_train_features_5 = output[24]['output']
        pos_test_data_5, pos_test_features_5 = output[25]['output']
        neg_test_data_5, neg_test_features_5 = output[26]['output']
        neu_test_data_5, neu_test_features_5 = output[27]['output']

        pos_train_data_6, pos_train_features_6 = output[28]['output']
        neg_train_data_6, neg_train_features_6 = output[29]['output']
        neu_train_data_6, neu_train_features_6 = output[30]['output']
        pos_test_data_6, pos_test_features_6 = output[31]['output']
        neg_test_data_6, neg_test_features_6 = output[32]['output']
        neu_test_data_6, neu_test_features_6 = output[33]['output']

        pos_train = pos_train_data_1 + pos_train_data_2 + pos_train_data_3 + pos_train_data_5 + pos_train_data_6 + pos_train_data_4
        neg_train = neg_train_data_1 + neg_train_data_2 + neg_train_data_3 + neg_train_data_5 + neg_train_data_6 + neg_train_data_4
        neu_train = neu_train_data_1 + neu_train_data_2 + neu_train_data_3 + neu_train_data_5 + neu_train_data_6

        pos_train_features = pos_train_features_1 + pos_train_features_2 + pos_train_features_3 + pos_train_features_5 + pos_train_features_6 + pos_train_features_4
        neg_train_features = neg_train_features_1 + neg_train_features_2 + neg_train_features_3 + neg_train_features_5 + neg_train_features_6 + neg_train_features_4
        neu_train_features = neu_train_features_1 + neu_train_features_2 + neu_train_features_3 + neu_train_features_5 + neu_train_features_6

        pos_test = pos_test_data_1 + pos_test_data_2 + pos_test_data_3 + pos_test_data_5 + pos_test_data_6 + pos_test_data_4
        neg_test = neg_test_data_1 + neg_test_data_2 + neg_test_data_3 + neg_test_data_5 + neg_test_data_6 + neg_test_data_4
        neu_test = neu_test_data_1 + neu_test_data_2 + neu_test_data_3 + neu_test_data_5 + neu_test_data_6

        pos_test_features = pos_test_features_1 + pos_test_features_2 + pos_test_features_3 + pos_test_features_5 + pos_test_features_6 + pos_test_features_4
        neg_test_features = neg_test_features_1 + neg_test_features_2 + neg_test_features_3 + neg_test_features_5 + neg_test_features_6 + neg_test_features_4
        neu_test_features = neu_test_features_1 + neu_test_features_2 + neu_test_features_3 + neu_test_features_5 + neu_test_features_6

        x_train = pos_train + neg_train + neu_train

        y_train = pos_train_features + neg_train_features + neu_train_features

        x_test = pos_test + neg_test + neu_test

        y_test = pos_test_features + neg_test_features + neu_test_features
    else:
        if check_neutral_existence:
            pos_train_data, pos_train_features = output[0]['output']
            neg_train_data, neg_train_features = output[1]['output']
            neu_train_data, neu_train_features = output[2]['output']
            pos_test_data, pos_test_features = output[3]['output']
            neg_test_data, neg_test_features = output[4]['output']
            neu_test_data, neu_test_features = output[5]['output']

            x_train = pos_train_data + neg_train_data + neu_train_data

            y_train = pos_train_features + neg_train_features + neu_train_features

            x_test = pos_test_data + neg_test_data + neu_test_data

            y_test = pos_test_features + neg_test_features + neu_test_features

        else:
            pos_train_data, pos_train_features = output[0]['output']
            neg_train_data, neg_train_features = output[1]['output']
            pos_test_data, pos_test_features = output[2]['output']
            neg_test_data, neg_test_features = output[3]['output']

            x_train = pos_train_data + neg_train_data
            y_train = pos_train_features + neg_train_features

            x_test = pos_test_data + neg_test_data

            y_test = pos_test_features + neg_test_features

    # printing data info
    print('train data info')
    print('train data size', len(x_train))
    print('------------------------------------')

    print('Test data info')
    print('test data size', len(x_test))
    print('------------------------------------')

    print('merging all features ... ')
    print('len(all_features):', len(y_train))
    print('------------------------------------')

    print('compute frequencies')
    all_features_freq = nltk.FreqDist(w for w in y_train)
    print('sample frequencies')
    print(all_features_freq.most_common(10))

    # Selecting the top 1% from the data
    ###################################
    # Selecting: top 1% of the top unique features
    my_features = list(all_features_freq)[:int(all_features_freq.B() * 0.1)]
    ###################################

    print(len(my_features), 'are kept out of', all_features_freq.B())
    print('features are selected')
    print('------------------------------------')

    print('generating features for training documents ...')
    feature_sets = [(document_features(d, my_features), c) for (d, c) in x_train]
    print('------------------------------------')
    print('training ...')
    # get the classfier
    clf = SklearnClassifier(
        classifier_based_on_number(classifier_number, x_train, y_train, x_test, y_test, dataset_number))

    # train the classifier for the training data
    classifier = clf.train(feature_sets)

    print('training is done')
    print('------------------------------------')
    print('generating features for test documents ...')
    test_features = [(document_features(d, my_features), c) for (d, c) in x_test]

    # testing the data
    ref_sets = collections.defaultdict(set)
    test_sets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(test_features):
        ref_sets[label].add(i)
        observed = classifier.classify(feats)
        test_sets[observed].add(i)

    print('test results:')
    # getting the precision, recall and f1-score for the tested data
    pos_pre = precision(ref_sets['P'], test_sets['P'])
    pos_rec = recall(ref_sets['P'], test_sets['P'])
    pos_f1 = f_measure(ref_sets['P'], test_sets['P'])

    neg_pre = precision(ref_sets['N'], test_sets['N'])
    neg_rec = recall(ref_sets['N'], test_sets['N'])
    neg_f1 = f_measure(ref_sets['N'], test_sets['N'])

    # if the data include the neutrals class
    if check_neutral_existence:
        neu_pre = precision(ref_sets['U'], test_sets['U'])
        neu_rec = recall(ref_sets['U'], test_sets['U'])
        neu_f1 = f_measure(ref_sets['U'], test_sets['U'])

        # calculate the weighted precision, recall and f1-score
        w_pre = (pos_pre * len(ref_sets['P']) + neg_pre * len(ref_sets['N']) + neu_pre * len(ref_sets['U'])) / (
                len(test_sets['N']) + len(test_sets['P']) + len(test_sets['U']))
        w_rec = (pos_rec * len(ref_sets['P']) + neg_rec * len(ref_sets['N']) + neu_rec * len(ref_sets['U'])) / (
                len(test_sets['N']) + len(test_sets['P']) + len(test_sets['U']))
        w_f1 = (pos_f1 * len(ref_sets['P']) + neg_f1 * len(ref_sets['N']) + neu_f1 * len(ref_sets['U'])) / (
                len(test_sets['N']) + len(test_sets['P']) + len(test_sets['U']))
        print('pos precision: ', pos_pre)
        print('pos recall:', pos_rec)

        print('neg precision: ', neg_pre)
        print('neg recall:', neg_rec)

        print('neu precision: ', neu_pre)
        print('neu recall:', neu_rec)

        print('positive f-score:', pos_f1)
        print('negative f-score:', neg_f1)
        print('neutral f-score:', neu_f1)
        print('########################## \n \n')
        print('accuracy: ', nltk.classify.accuracy(classifier, test_features))
        print('weighted precision = ', w_pre)
        print('weighted recall = ', w_rec)
        print('weighted F1 = ', w_f1)
    else:
        w_pre = (pos_pre * len(ref_sets['P']) + neg_pre * len(ref_sets['N'])) / (
                len(test_sets['N']) + len(test_sets['P']))
        w_rec = (pos_rec * len(ref_sets['P']) + neg_rec * len(ref_sets['N'])) / (
                len(test_sets['N']) + len(test_sets['P']) + len(test_sets['U']))
        w_f1 = (pos_f1 * len(ref_sets['P']) + neg_f1 * len(ref_sets['N'])) / (
                len(test_sets['N']) + len(test_sets['P']))

        print('pos precision: ', pos_pre)
        print('pos recall:', pos_rec)

        print('neg precision: ', neg_pre)
        print('neg recall:', neg_rec)

        print('positive f-score:', pos_f1)
        print('negative f-score:', neg_f1)
        print('########################## \n \n')
        print('accuracy: ', nltk.classify.accuracy(classifier, test_features))
        print('weighted precision = ', w_pre)
        print('weighted recall = ', w_rec)
        print('weighted F1 = ', w_f1)


# save the results of the classifier in a file in the results folder
def save_sa_results(dataset, stemmer, classifier, ngrams_list):
    # for each gram in the n-gram list
    for i in range(0, len(ngrams_list)):
        # the file name depending on the n-gram number with the dataset, stemmer and classifier number
        outfile = 'results/Classification using ' + str(ngrams_list[i]) + '-grams for ' + str(dataset) + '_' + str(
            stemmer) + '_' + str(
            classifier) + '.result'
        # open the file we want to write on
        sys.stdout = open(outfile, mode='w', encoding='utf-8')
        # timer for calculate the time needed to do the classification
        start = timer()
        print('classification using ' + str(ngrams_list[i]) + '-grams for ' + str(dataset) + '_' + str(
            stemmer) + '_' + str(
            classifier))
        # do the classification
        # each print after we opened the file is executed on the file not on the console
        SA_using_ngrams(1, 4, 1, n_grams[i])
        end = timer()
        # we print the time took for the classifier
        print('time taking for training the classifier: ' + str(end - start))
        sys.stdout.close()


# when we need to classify we need to have it in the main
# because we used parallel preprocessing for the data
if __name__ == '__main__':
    # we choose the desired dataset, stemmer, classifier number and the list of ngrams
    # we want to build the model on it.
    dataset = 1
    stemmer = 4
    classifier = 1
    n_grams = [1, 2]
    save_sa_results(dataset, stemmer, classifier, n_grams)
