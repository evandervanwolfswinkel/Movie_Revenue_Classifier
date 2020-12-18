from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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
revenueclass = []
with open('./data/movies_metadata.csv', newline='', encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['budget'] != "0" and row['revenue'] != "0" and RepresentsInt(row['budget']) and row['runtime'] != "":
            budget = float(row['budget'])
            revenue = float(row['revenue'])
            runtime = float(row['runtime'])

            moviesdata.append([budget,runtime,row['original_language']])
            revenueclass.append(revenue)

print(moviesdata)
print(revenueclass)
print(len(moviesdata))
print(len(revenueclass))


X_train, X_test, y_train, y_test = train_test_split(moviesdata, revenueclass, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
pred = clf.predict_proba(X_test[:1])
print(pred)
