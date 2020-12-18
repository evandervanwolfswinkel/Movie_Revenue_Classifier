from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
import numpy as np
import csv

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

            profit = revenue - budget

            if profit > 1:
                profit = 1
            else:
                profit = 0
            runtime = float(row['runtime'])
            language = str(row['original_language'])


            moviesdata.append([budget,runtime,language])
            profitclass.append(profit)

moviesdata = DataFrame(moviesdata)
profitclass = DataFrame(profitclass)

print(moviesdata)
print(profitclass)
print(moviesdata.shape)
print(profitclass.shape)

ordinal = OrdinalEncoder()
X = ordinal.fit_transform(moviesdata)
print(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(np.ravel(profitclass))
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

pred = clf.predict_proba(X_test[:1])
print(pred)
