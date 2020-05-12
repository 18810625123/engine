# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = pd.read_csv('./data.csv')
# Assign the features to the variable X, and the labels to the variable y.
X = data[['x1','x2']].values
y = data['y'].values
# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.

def aa(a,b):
    # model = DecisionTreeClassifier(min_samples_leaf=a, min_samples_split=b)
    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators=a)
    model.fit(X,y)
    # TODO: Fit the model.
    r = model.predict(X)
    print('%s %s\t %s' % (a, b, accuracy_score(r, y)))

for i in range(1, 20):
    aa(i, 2)

