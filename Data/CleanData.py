
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score
from sklearn import ensemble
from sklearn import tree
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt





data_folder = "../Data"
train_file = "/adult.data.txt"
test_file = "/adult.test.txt"
cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
        "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

train_df = pd.read_csv(data_folder + train_file, names=cols, header=None)
test_df  = pd.read_csv(data_folder + test_file, names=cols, skiprows=1)


#print(train_df.ix[:,1])
#print(train_df['class'])
print
print
print
print
print
# One Hot Encoding
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
train_df = pd.get_dummies(train_df, columns=categorical_cols)
test_df = pd.get_dummies(test_df, columns=categorical_cols)

#print(train_df[train_df.categorical_cols[0]])
#print(train_df.sort_index(axis=1, ascending=False))
#print(train_df[[col for col in train_df if col.startswith('workclass_')]])



# convert class to 0 or 1
train_df["class"] = train_df["class"].astype('category')
train_df["class"] = train_df["class"].cat.codes
test_df["class"]  = test_df["class"].astype('category')
test_df["class"]  = test_df["class"].cat.codes

X_train = train_df.drop("class", axis=1).as_matrix()
y_train = train_df["class"].as_matrix()


X_test = test_df.drop("class", axis=1).as_matrix()
y_test = test_df["class"].as_matrix()


# clf = linear_model.RidgeClassifier()
# clf.fit(X_train, y_train)
# n_folds = 10
# scores = cross_val_score(clf, X_train, y_train, cv=n_folds)
# print(scores)
#
# y_pred = cross_val_predict(clf, X_test, y_test, cv=n_folds)
# print(accuracy_score(y_test, y_pred))

#n_folds = 10
#kf = KFold(n_folds, random_state=None, shuffle=False)
#functionPred = cross_val_score()
#accuracyPreds = cross_val_score(linear_model.LogisticRegression(),
#                                  X_test, y_test, cv = kf, scoring="accuracy")
#print(accuracyPreds.mean()) ~80%


#TREE STUFF
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
# preds = clf.predict(X_test)
# res = (accuracy_score(y_test, preds))
# print(res)

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
importance = clf.feature_importances_

pos = np.arange(8) + 0.5
plt.barh(pos, importance, align='center')






# data = pd.read_csv('adult.data.txt', sep=" ", header=None)
# data.columns = ["age", "workClass", "fnlwgt", "eduction", "education-num",
#                 "marital-status", "occupation", "relationship", "race",
#                 "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "Salary"]
#print(data.iloc[10000])
#print(data[['age']])
#print (data[ pd.to_numeric(data['age'], errors='coerce').isnull()])
#data['age'] = pd.to_numeric(data['age'], errors='coerce')
#data['age'] = data['age'].astype(float)

#print(data.head())

#data.hist(column = 'age', bins=50)
#plt.show()
#data = data.fillna(np.nan)
#print(data)
