from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
from pandas import options
from pandas import crosstab
import numpy as np
import csv
from sklearn import tree

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

moviesdata = []
profitclass = []
with open('./data/movies_metadata.csv', newline='', encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['budget'] != "0" and row['revenue'] != "0" and RepresentsInt(row['budget']) and row['runtime'] != "":

            budget = float(row['budget'])
            revenue = float(row['revenue'])
            runtime = float(row['runtime'])
            language = str(row['original_language'])
            vote_average = float(row['vote_average'])
            profit = revenue - budget
            #
            # if budget > 200000000:
            #     budget = "Very High"
            # elif budget > 100000000:
            #     budget = "High"
            # elif budget > 31000000:
            #     budget = "Above average"
            # elif budget > 2000000:
            #     budget = "Below average"
            # elif budget <= 2000000:
            #     budget = "Low"
            #
            # if runtime > 300:
            #     runtime = "Very High"
            # elif runtime > 200:
            #     runtime = "High"
            # elif runtime > 110:
            #     runtime = "Above average"
            # elif runtime > 60:
            #     runtime = "Below average"
            # elif runtime <= 60:
            #     runtime = "Low"
            #
            # if vote_average > 9:
            #     vote_average = "Very High"
            # elif vote_average > 7.5:
            #     vote_average = "High"
            # elif vote_average > 6:
            #     vote_average = "Above average"
            # elif vote_average > 4:
            #     vote_average = "Below average"
            # elif vote_average <= 4:
            #     vote_average = "Low"

            if profit > 10000000:
                profit = "Profitable"
            else:
                profit = "Not Profitable"

            moviesdata.append([budget,runtime,language,vote_average])
            profitclass.append(profit)

# print(sum(list(map(itemgetter(3), moviesdata))) / len(moviesdata) )
# print(max(list(map(itemgetter(3), moviesdata))))
# print(min(list(map(itemgetter(3), moviesdata))))

moviesdata = DataFrame(moviesdata)
profitclass = DataFrame(profitclass)

# print(moviesdata)
# print(profitclass)
# print(moviesdata.shape)
# print(profitclass.shape)

ordinal = OrdinalEncoder()
X = ordinal.fit_transform(moviesdata)
print(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(np.ravel(profitclass))
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.10)

clf = MLPClassifier(hidden_layer_sizes=(5,1000,100,5,),random_state=1, max_iter=3000,learning_rate_init=0.001).fit(X_train, y_train)

pred = clf.score(X_test, y_test)
print("Neural Network Score: ", pred, "\n")

y_test_predicted = clf.predict(X_test)

df_confusion = crosstab(y_test, y_test_predicted, rownames=['Actual'], colnames=['Predicted'], margins=True)
print("Neural Network Confusion Matrix: ")
print(df_confusion)
print("\n")

# Decision tree just with scikit learn
config = {'algorithm': 'C4.5'}
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X_train, y_train)

pred_dt = classifier.score(X_test, y_test)
print("Decision Tree Score: ", pred_dt, "\n")

y_test_predicted_dt = classifier.predict(X_test)
df_confusion_dt = crosstab(y_test, y_test_predicted_dt, rownames=['Actual'], colnames=['Predicted'], margins=True)
print("Decision Tree Confusion Matrix: ")
print(df_confusion_dt)

