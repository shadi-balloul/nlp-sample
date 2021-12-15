import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime


"""""
print ('the number of tweets')

objects = ('Positive', 'Negative', 'Objective')
y_pos = np.arange(len(objects))
#performance = [3000,28,2000]

#merged dataset
performance = [141977 ,133362 ,17058]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number Of Tweets')
plt.xlabel('Tweets Classes')

plt.savefig('charts/dataset7.png', dpi=600, bbox_inches='tight')
"""""

###########################################################
############################################################
################# Merged Dataset
#########################################################
##############################################################
database = 7
stemmer = 4
clf = 1
databases = json.load(open("Objects/databases.json"))
stemmers = json.load(open('Objects/stemmers.json'))
clfs = json.load(open('Objects/classifiers.json'))

filePath = 'results/res-' + str(database) + '_' + str(stemmer) + '_' + str(clf) + '.json'
res = pd.read_json(filePath)
result = pd.DataFrame(columns=res.columns)

for clf in range(1, 9):
    if stemmer != 5:
        filePath = 'results/cr-' + str(database) + '_' + str(stemmer) + '_' + str(clf) + '.json'
        print(filePath)
        res = pd.read_json(filePath)
        result = result.append(res)

result = result.astype(float)
result = result.round(3)
print("result.....")
#result.index = result.index + result.groupby(level=0).cumcount().astype(str)
print(result.index)
print(result)
precision = result.iloc[::4, 5]
recall = result.iloc[1::4, 5]
f1score = result.iloc[2::4, 5]
accuracy = result.iloc[::4, 3]
print(precision)
print(recall)
print('f1score...')
print(f1score)





x = np.arange(1.0, 9.0, 1.0)
y = x + 0.2
z = x + 0.4
acc = x + 0.6

print(x)

ax = plt.subplot(111)
ax.bar(x, recall, width=0.2, color='b', align='center', label='Recall Weighted Avg')
ax.bar(y, precision, width=0.2, color='g', align='center',  label='Precision Weighted Avg')
ax.bar(z, f1score, width=0.2, color='r', align='center', label='F-Score')
ax.bar(acc, f1score, width=0.2, color='y', align='center', label='Accuracy')
ax.set_xticks(y)
ax.set_xticklabels(("Linear\n SVC", "SVC", "SGD", "Multinomial\n NB", "Bernoulli\n NB", "Decision\n Trees", "Random Forest\n Trees", "KNN"), rotation=60)
#ax.legend(loc='upper center', bbox_to_anchor=(0.485, -0.05),
#          fancybox=True, ncol=3)

ax.legend(loc='upper center',  bbox_to_anchor=(0.5, 1.2), fancybox=True, ncol=3)
#ax.title.set_text('Stemmers evaluation with the Merged dataset and Multinomial Naive Bayes Classifier')

#for index,data in enumerate(recall):
#    plt.text(x=index + 1 , y =data+0.1 , s=f"{data}" , fontdict=dict(fontsize=10))

fig1 = plt.gcf()
fig1.show()
fig1.savefig("charts/d7.png", bbox_inches='tight', dpi=600)



precision_p = result.iloc[::4, 0]
precision_n = result.iloc[::4, 1]
precision_u = result.iloc[::4, 2]

ax = plt.subplot(111)
ax.bar(x, precision_p, width=0.2, color='b', align='center', label='Positive')
ax.bar(y, precision_n, width=0.2, color='g', align='center',  label='Negative')
ax.bar(z, precision_u, width=0.2, color='r', align='center', label='Neutral')
ax.set_xticks(y)
ax.set_xticklabels(("Linear\n SVC", "SVC", "SGD", "Multinomial\n NB", "Bernoulli\n NB", "Decision\n Trees", "Random Forest\n Trees", "KNN"), rotation=60)
ax.set_ylabel("Precision")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, ncol=3)
# ax.title.set_text('The precision of each class in all stemmers \n in case of Linear SVC and the merged dataset')

fig2 = plt.gcf()
fig2.show()
fig2.savefig("charts/d7-prec.png", bbox_inches='tight', dpi=600)


##########################################################
############################################################
################# End Merged Dataset
#########################################################
##############################################################




"""""
################################
###############################
#The Motaz Dataset1
#########################
#########################
database = 6
stemmer = 4
clf = 1
databases = json.load(open("Objects/databases.json"))
stemmers = json.load(open('Objects/stemmers.json'))
clfs = json.load(open('Objects/classifiers.json'))

filePath = 'results/res-' + str(database) + '_' + str(stemmer) + '_' + str(clf) + '.json'
res = pd.read_json(filePath)
result = pd.DataFrame(columns = res.columns)

for clf in range(1, 9):
    if stemmer != 5:
        filePath = 'results/cr-' + str(database) + '_' + str(stemmer) + '_' + str(clf) + '.json'
        print(filePath)
        res = pd.read_json(filePath)
        result = result.append(res)

result = result.astype(float)
result = result.round(3)
print("result.....")
#result.index = result.index + result.groupby(level=0).cumcount().astype(str)
print(result.index)
print(result)
precision = result.iloc[::4, 5]
recall = result.iloc[1::4, 5]
f1score = result.iloc[2::4, 5]
print(precision)
print(recall)
print('f1score...')
print(f1score)

#result.to_csv("eval/7-stemmers-1.csv")

from matplotlib.dates import date2num
import datetime

x = np.arange(1.0, 9.0, 1.0)
y = x + 0.2
z = x + 0.4

print(x)

ax = plt.subplot(111)
ax.bar(x, recall, width=0.2, color='b', align='center', label='Recall Weighted Avg')
ax.bar(y, precision, width=0.2, color='g', align='center',  label='Precision Weighted Avg')
ax.bar(z, f1score, width=0.2, color='r', align='center', label='F-Score')
ax.set_xticks(y)
ax.set_xticklabels(("Linear\n SVC", "SVC", "SGD", "Multinomial\n NB", "Bernoulli\n NB", "Decision\n Trees", "Random Forest\n Trees", "KNN"), rotation=60)
#ax.legend(loc='upper center', bbox_to_anchor=(0.485, -0.05),
#          fancybox=True, ncol=3)

ax.legend(loc='upper center',  bbox_to_anchor=(0.5, 1.1), fancybox=True, ncol=3)
#ax.title.set_text('Stemmers evaluation with the Merged dataset and Multinomial Naive Bayes Classifier')

#for index,data in enumerate(recall):
#    plt.text(x=index + 1 , y =data+0.1 , s=f"{data}" , fontdict=dict(fontsize=10))

fig1 = plt.gcf()
fig1.show()
fig1.savefig("charts/d6.png", bbox_inches='tight', dpi=600)



precision_p = result.iloc[::4, 0]
precision_n = result.iloc[::4, 1]
precision_u = result.iloc[::4, 2]

ax = plt.subplot(111)
ax.bar(x, precision_p, width=0.2, color='b', align='center', label='Positive')
ax.bar(y, precision_n, width=0.2, color='g', align='center',  label='Negative')
ax.bar(z, precision_u, width=0.2, color='r', align='center', label='Neutral')
ax.set_xticks(y)
ax.set_xticklabels(("Linear\n SVC", "SVC", "SGD", "Multinomial\n NB", "Bernoulli\n NB", "Decision\n Trees", "Random Forest\n Trees", "KNN"), rotation=60)
ax.set_ylabel("Precision")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, ncol=3)
# ax.title.set_text('The precision of each class in all stemmers \n in case of Linear SVC and the merged dataset')

fig2 = plt.gcf()
fig2.show()
fig2.savefig("charts/d6-prec.png", bbox_inches='tight', dpi=600)

################################
###############################
#End  Motaz Dataset
#########################
#########################
"""""





"""""
################################
###############################
The second Dataset
#########################
#########################
database = 3
stemmer = 4
clf = 1
databases = json.load(open("Objects/databases.json"))
stemmers = json.load(open('Objects/stemmers.json'))
clfs = json.load(open('Objects/classifiers.json'))

filePath = 'results/res-' + str(database) + '_' + str(stemmer) + '_' + str(clf) + '.json'
res = pd.read_json(filePath)
result = pd.DataFrame(columns = res.columns)

for clf in range(1, 9):
    if stemmer != 5:
        filePath = 'results/cr-' + str(database) + '_' + str(stemmer) + '_' + str(clf) + '.json'
        print(filePath)
        res = pd.read_json(filePath)
        result = result.append(res)

result = result.astype(float)
result = result.round(3)
print("result.....")
#result.index = result.index + result.groupby(level=0).cumcount().astype(str)
print(result.index)
print(result)
precision = result.iloc[::4, 5]
recall = result.iloc[1::4, 5]
f1score = result.iloc[2::4, 5]
print(precision)
print(recall)
print('f1score...')
print(f1score)

#result.to_csv("eval/7-stemmers-1.csv")

from matplotlib.dates import date2num
import datetime

x = np.arange(1.0, 9.0, 1.0)
y = x + 0.2
z = x + 0.4

print(x)

ax = plt.subplot(111)
ax.bar(x, recall, width=0.2, color='b', align='center', label='Recall Weighted Avg')
ax.bar(y, precision, width=0.2, color='g', align='center',  label='Precision Weighted Avg')
ax.bar(z, f1score, width=0.2, color='r', align='center', label='F-Score')
ax.set_xticks(y)
ax.set_xticklabels(("Linear\n SVC", "SVC", "SGD", "Multinomial\n NB", "Bernoulli\n NB", "Decision\n Trees", "Random Forest\n Trees", "KNN"), rotation=60)
#ax.legend(loc='upper center', bbox_to_anchor=(0.485, -0.05),
#          fancybox=True, ncol=3)

ax.legend(loc='upper center',  bbox_to_anchor=(0.5, 1.08), ncol=3)
#ax.title.set_text('Stemmers evaluation with the Merged dataset and Multinomial Naive Bayes Classifier')

#for index,data in enumerate(recall):
#    plt.text(x=index + 1 , y =data+0.1 , s=f"{data}" , fontdict=dict(fontsize=10))

fig1 = plt.gcf()
fig1.show()
fig1.savefig("charts/d2.png", bbox_inches='tight', dpi=600)
"""""