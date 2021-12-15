import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

# datetime object containing current date and time
now = datetime.now()

print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

database = 7
stemmer = 1
clf = 1
databases = json.load(open("Objects/databases.json"))
stemmers = json.load(open('Objects/stemmers.json'))
clfs = json.load(open('Objects/classifiers.json'))

filePath = 'results/res-' + str(database) + '_' + str(stemmer) + '_' + str(clf) + '.json'
res = pd.read_json(filePath)
result = pd.DataFrame(columns = res.columns)

for stemmer in range(1, 7):
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
f1_score = result.iloc[2::4, 5]
print(precision)
print(recall)
print(recall.T)

#result.to_csv("eval/7-stemmers-1.csv")

from matplotlib.dates import date2num
import datetime

x = np.arange(1.0, 6.0, 1.0)
y = x + 0.2
z = x + 0.4

print(x)

ax = plt.subplot(111)
ax.bar(x, recall, width=0.2, color='b', align='center', label='Recall Weighted Avg')
ax.bar(y, precision, width=0.2, color='g', align='center',  label='precision Weighted Avg')
ax.bar(z, f1_score, width=0.2, color='r', align='center', label='F1-Score Weighted Avg')
ax.set_xticks(y)
ax.set_xticklabels(("COG", "Snowball", "ISRI", "Improved ISRI", "Tashaphyne"))
ax.legend(loc='upper center', bbox_to_anchor=(0.485, -0.05),
          fancybox=True, ncol=5)
ax.title.set_text('Stemmers evaluation with the Merged dataset and Multinomial Naive Bayes Classifier')

#for index,data in enumerate(recall):
#    plt.text(x=index + 1 , y =data+0.1 , s=f"{data}" , fontdict=dict(fontsize=10))

fig1 = plt.gcf()
fig1.show()
fig1.savefig("charts/7-stemmers-4.png", bbox_inches='tight')

"""""
precision_p = result.iloc[::4, 0]
precision_n = result.iloc[::4, 1]
precision_u = result.iloc[::4, 2]

ax = plt.subplot(111)
ax.bar(x, precision_p, width=0.2, color='b', align='center', label='Positive')
ax.bar(y, precision_n, width=0.2, color='g', align='center',  label='Negative')
ax.bar(z, precision_u, width=0.2, color='r', align='center', label='Neutral')
ax.set_xticks(y)
ax.set_xticklabels(("COG", "Snowball", "ISRI", "Improved ISRI", "Tashaphyne"))
ax.set_ylabel("Precision")
ax.legend(loc='upper center', bbox_to_anchor=(0.485, -0.05),
          fancybox=True, ncol=5)
# ax.title.set_text('The precision of each class in all stemmers \n in case of Linear SVC and the merged dataset')

fig2 = plt.gcf()
fig2.show()
fig2.savefig("charts/stem71-prec.png", bbox_inches='tight')

#ax2 = plt.subplot(1)
accuracy = result.iloc[::4, 3]
plt.bar(("COG", "Snowball", "ISRI", "Improved ISRI", "Tashaphyne"), accuracy, color='blue')
#plt.()
#plt.xlabel("Energy Source")
#plt.ylabel("Energy Output (GJ)")
plt.title("The Accuracy of models with all stemmers, merged dataset \n and SVCLinear Classifier")

#ax2 = plt.subplot(111)
#ax2.bar(x, precision_p, width=0.2, color='b', align='center', label='Positive')
#ax2.set_xticks(x)
#ax2.set_xticklabels(("COG", "Snowball", "ISRI", "Improved ISRI", "Tashaphyne"))

fig3 = plt.gcf()
fig3.show()
fig3.savefig("charts/stem71-acc.png", bbox_inches='tight')

"""""
