import glob
import re
from csv import writer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from PreProcessing import normalize_arabic, remove_diacritics, remove_punctuations


# we used this function for saving the tweets catched by Streamlistener
def AddTweetToDataset(tweet, file_name):
    tweets = list()
    tweets.append(tweet)
    # Open file in append mode
    with open(file_name, 'a+', newline='', encoding="utf-8") as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(tweets)


# split the file into 3 files one for each class
def split_the_data_into_PNU_csv(csv_file, contains_neutrals):
    # if the database contains neutrals tweets
    if contains_neutrals:
        # get the dataset name from the file name
        dataset_name = csv_file.split('.')[0]
        df = pd.read_csv(csv_file, encoding='UTF-8')
        # prepare the dataframes that we will add the data on it
        df_p = pd.DataFrame()
        df_n = pd.DataFrame()
        df_u = pd.DataFrame()
        # indicate the row number
        index = 0

        # we add the tweets into the corresponded dataframe depending on the classification type
        for row in df['Classification']:
            if row == "P":
                df_p = df_p.append({'Tweet': df.Tweet[index]}, ignore_index=True)
                index += 1
            elif row == "N":
                df_n = df_n.append({'Tweet': df.Tweet[index]}, ignore_index=True)
                index += 1
            else:
                df_u = df_u.append({'Tweet': df.Tweet[index]}, ignore_index=True)
                index += 1

        nan_value = float("NaN")

        # deleting the NaN rows which are the empty rows in our dataframes
        df_p.replace("", nan_value, inplace=True)
        df_p.dropna(subset=["Tweet"], inplace=True)
        df_n.replace("", nan_value, inplace=True)
        df_n.dropna(subset=["Tweet"], inplace=True)
        df_u.replace("", nan_value, inplace=True)
        df_u.dropna(subset=["Tweet"], inplace=True)

        # saving the dataframes in CSVs
        pos_file = str(dataset_name) + '_pos_tweets.csv'
        neg_file = str(dataset_name) + '_neg_tweets.csv'
        neu_file = str(dataset_name) + '_neu_tweets.csv'
        df_p.to_csv(pos_file, index=False, encoding='UTF-8')
        df_n.to_csv(neg_file, index=False, encoding='UTF-8')
        df_u.to_csv(neu_file, index=False, encoding='UTF-8')
        return pos_file, neg_file, neu_file
    else:
        # if the dataset dose not include neutrals tweets we do the same process for the positive and negative tweets
        dataset_name = csv_file.split('.')[0]
        df = pd.read_csv(csv_file, encoding='UTF-8')
        df_p = pd.DataFrame()
        df_n = pd.DataFrame()
        index = 0
        for row in df['Classification']:
            if row == "P":
                df_p = df_p.append({'Tweet': df.Tweet[index]}, ignore_index=True)
                index += 1
            elif row == "N":
                df_n = df_n.append({'Tweet': df.Tweet[index]}, ignore_index=True)
                index += 1
            else:
                index += 1
        nan_value = float("NaN")
        df_p.replace("", nan_value, inplace=True)
        df_p.dropna(subset=["Tweet"], inplace=True)
        df_n.replace("", nan_value, inplace=True)
        df_n.dropna(subset=["Tweet"], inplace=True)

        pos_file = str(dataset_name) + '_pos_tweets.csv'
        neg_file = str(dataset_name) + '_neg_tweets.csv'

        df_p.to_csv(pos_file, index=False, encoding='UTF-8')
        df_n.to_csv(neg_file, index=False, encoding='UTF-8')
        return pos_file, neg_file


# split the file for a training file and a testing file based on ratio
def split_training_testing(file, ratio):
    df = pd.read_csv(file, encoding='UTF-8')
    train, test = train_test_split(df, test_size=ratio, random_state=42, shuffle=True)
    return train, test


# the number of the new dataset to be saved in the Data file
def get_the_next_dataset_number():
    list_of_files = glob.glob('Data/*')
    results = dict()
    for i in range(0, len(list_of_files)):
        file = list_of_files[i]
        temp = file.split('\\')[1]
        number = int(temp.split('_')[0])
        results[number] = results.get(number, 0) + 1

    return max(results, key=lambda key: results[key]) + 1


# splitting the three files for each class into a training and a testing file based on ratio
def save_splitting_csv(csv_file, contains_neutrals, ratio):
    dataset_number = get_the_next_dataset_number()
    if contains_neutrals:
        pos_file, neg_file, neu_file = split_the_data_into_PNU_csv(csv_file, contains_neutrals)
        train_pos, test_pos = split_training_testing(pos_file, ratio)
        train_neg, test_neg = split_training_testing(neg_file, ratio)
        train_neu, test_neu = split_training_testing(neu_file, ratio)

        train_pos.to_csv('Data/' + str(dataset_number) + '_train_pos.csv', encoding='UTF-8', index=False)
        test_pos.to_csv('Data/' + str(dataset_number) + '_test_pos.csv', encoding='UTF-8', index=False)
        train_neg.to_csv('Data/' + str(dataset_number) + '_train_neg.csv', encoding='UTF-8', index=False)
        test_neg.to_csv('Data/' + str(dataset_number) + '_test_neg.csv', encoding='UTF-8', index=False)
        train_neu.to_csv('Data/' + str(dataset_number) + '_train_neu.csv', encoding='UTF-8', index=False)
        test_neu.to_csv('Data/' + str(dataset_number) + '_test_neu.csv', encoding='UTF-8', index=False)

    else:
        pos_file, neg_file = split_the_data_into_PNU_csv(csv_file, contains_neutrals)
        train_pos, test_pos = split_training_testing(pos_file, ratio)
        train_neg, test_neg = split_training_testing(neg_file, ratio)

        train_pos.to_csv('Data/' + str(dataset_number) + '_train_pos.csv', encoding='UTF-8', index=False)
        test_pos.to_csv('Data/' + str(dataset_number) + '_test_pos.csv', encoding='UTF-8', index=False)
        train_neg.to_csv('Data/' + str(dataset_number) + '_train_neg.csv', encoding='UTF-8', index=False)
        test_neg.to_csv('Data/' + str(dataset_number) + '_test_neg.csv', encoding='UTF-8', index=False)


# merging the our build stopwords with nltk stopwords
def get_all_stopwords():
    with open("Objects/stopwords.txt", "r", encoding='UTF-8') as stopwordsfile:
        stop_words = list(set(stopwords.words('arabic')))
        for row in stopwordsfile:
            row = re.sub('\n', "", row)
            if row not in stop_words:
                stop_words.append(row)

    i = 0
    for row in stop_words:
        row = normalize_arabic(row)
        row = remove_diacritics(row)
        row = remove_punctuations(row)
        stop_words[i] = row
        i += 1
    return list(dict.fromkeys(stop_words))


# saving the stopwords file in sw.txt
def save_stopwords_file():
    sw = get_all_stopwords()
    with open('Objects/sw.txt', 'w', encoding='UTF-8') as f:
        for ele in sw:
            ele = re.sub(r" ", "", ele)
            f.write(ele + '\n')


