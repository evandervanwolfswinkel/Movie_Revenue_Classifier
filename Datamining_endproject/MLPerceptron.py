from operator import itemgetter

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.metrics import plot_confusion_matrix
from pandas import DataFrame
from pandas import options
from pandas import crosstab
import numpy as np
import csv
import ast
import matplotlib.pyplot as plt

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

def feature_engineering(series):
    # Feature engineering for categorical data
    string_list = []
    for i in series:
        string_list.append(i)
    return LabelEncoder().fit_transform(string_list)

moviesdata = []
profitclass = []
with open('./data/movies_metadata.csv', newline='', encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        genrelistdict = ast.literal_eval(row['genres'][0:])
        try:
            genre = genrelistdict[0]['name']
        except:
            genre = ""

        try:
            prodcountrylistdict = ast.literal_eval(row['production_countries'][0:])
            prodcountry = prodcountrylistdict[0]['name']
        except:
            prodcountry = ""

        try:
            prodcomplistdict = ast.literal_eval(row['production_companies'][0:])
            prodcomp = prodcomplistdict[0]['name']
        except:
            prodcomp = ""

        if RepresentsInt(row['budget']) and float(row['budget']) > 100 and float(row['revenue']) > 100 and row['runtime'] != "" and genre != "" and prodcountry != "" and prodcomp != "":

            budget = float(row['budget'])
            revenue = float(row['revenue'])
            runtime = float(row['runtime'])
            language = str(row['original_language'])
            vote_average = float(row['vote_average'])
            returns = revenue / budget


            if returns > 4:
                profit = "High profit"
            elif returns > 1.2:
                profit = "Profit"
            else:
                profit = "Not Profitable"

            # if returns > 2:
            #     profit = "Profitable"
            # else:
            #     profit = "Not Profitable"

            moviesdata.append([budget,runtime,language,vote_average,genre,prodcountry,prodcomp])
            profitclass.append(profit)

# print(profitclass)
# print(sum(profitclass) / len(profitclass) )
# print(max(profitclass))
# print(min(profitclass))

moviesdata = DataFrame(moviesdata,columns=["budget","runtime","language","vote_average","genre","prodcountry","prodcomp"])
profitclass = DataFrame(profitclass,columns=["profit"])

# print(profitclass['profit'].describe().apply(lambda x: format(x, 'f')))

print(moviesdata)
# print(moviesdata)
# print(profitclass)
# print(moviesdata.shape)
# print(profitclass.shape)
scaler = StandardScaler()

moviesdata["prodcountry"] = feature_engineering(moviesdata['prodcountry'])
moviesdata["prodcomp"] = feature_engineering(moviesdata['prodcomp'])
moviesdata["language"] = feature_engineering(moviesdata['language'])
moviesdata["genre"] = feature_engineering(moviesdata['genre'])

print(moviesdata)
X = scaler.fit_transform(moviesdata)
print(X)

y = np.ravel(profitclass)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.30)
# print(X_train)
# X_train = scale(X_train)
# print(X_train)
# X_test = scale(X_test)

# clf = MLPClassifier(hidden_layer_sizes=(100,),random_state=1, max_iter=30000,
#                         learning_rate_init=0.001, solver='adam', activation='relu').fit(X_train, y_train)
# pred = clf.score(X_test, y_test)
# print(pred)

average_score = []
for i in range(6):
    clf = MLPClassifier(hidden_layer_sizes=(100,),random_state=i, max_iter=30000,
                        learning_rate_init=0.001, solver='adam', activation='relu').fit(
        X_train, y_train)


    pred = clf.score(X_test, y_test)
    print(pred)
    average_score.append(pred)

print("Average prediction score:")
print(sum(average_score)/len(average_score))


disp = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')

disp.ax_.set_title("Confusion matrix Movie Profitability Classifier")

print("Confusion matrix Movie Profitability Classifier")
print(disp.confusion_matrix)
plt.show()