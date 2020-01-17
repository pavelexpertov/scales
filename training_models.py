import pandas as pd
import sklearn

# In [ ]
df = pd.read_csv('balance-scale.data', header=None, names=['C','LW','LD','RW','RD'])
df

# In [ ]
# Creating a list of records where the format is ['description', mean, std]
recording_list = []
# In [ ]
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X = df.loc[:, ['LW','LD','RW','RD']]
y = df.loc[:, ['C']]

clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=3)
print('With 3 cross_validation', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With 3 cross_validation', scores.mean(), scores.std()])

clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=5)
print('With 5 cross_validation', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With 5 cross_validation', scores.mean(), scores.std()])

clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With 10 cross_validation', scores.mean(), scores.std()])


# <markdown>
# It looks like that the mean of cross validation results is affected by the number of specified folds during cross validation. It means that the decision tree model gets more accurate as you feed it a lot more data.

# Let's look at the importance of the features.

# In [ ]
clf = DecisionTreeClassifier()
clf.fit(X, y)
clf.feature_importances_

# <markdown>
# It looks like that importances of the attributes are balanced out well with a difference between 1 to 3 percent. Thus it means that pretty much they are equal to each other.

# Even though the balanced class (i.e. 'B') is only 8% of the dataset, models still looks like to struggle to perform good on average even though the it would have trained on a large number of 'L' or 'R' classes.

# I need to look at other machine learning algorithms to see whether they perform better.

# In [ ]
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
enc_y = LabelEncoder()
enc_y.fit(np.ravel(y.to_numpy()))
encoded_y = enc_y.transform(np.ravel(y.to_numpy()))
clf = LogisticRegression(random_state=0)
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With LogisticRegression', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With LogisticRegression', scores.mean(), scores.std()])

# In [ ]
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1000, tol=1e-3)
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With SGDClassifier', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With SGDClassifier', scores.mean(), scores.std()])

# In [ ]
from sklearn.svm import SVC
clf = SVC(gamma='auto')
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With SVC', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With SVC', scores.mean(), scores.std()])

# In [ ]
from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=5000)
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With LinearSVC', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With LinearSVC', scores.mean(), scores.std()])

# In [ ]
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200)
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With RandomForestClassifier', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With RandomForestClassifier', scores.mean(), scores.std()])

# In [ ]
from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB()
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With CategoricalNB', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With CategoricalNB', scores.mean(), scores.std()])

# In [ ]
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With NearestCentroid', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With NearestCentroid', scores.mean(), scores.std()])

# In [ ]
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With KNeighborsClassifier', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With KNeighborsClassifier', scores.mean(), scores.std()])

# In [ ]
from sklearn.neighbors import RadiusNeighborsClassifier
clf = RadiusNeighborsClassifier(radius=2.0)
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With RadiusNeighborsClassifier', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With RadiusNeighborsClassifier', scores.mean(), scores.std()])

# In [ ]
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(max_iter=1200)
scores = cross_val_score(clf, X, encoded_y, cv=10)
print('With MLPClassifier', 'Mean:', scores.mean(), scores.std())
recording_list.append(['With MLPClassifier', scores.mean(), scores.std()])

# In [ ]
sorted_list = sorted(recording_list, key=lambda item: item[1], reverse=True)
for classifier, mean, std in sorted_list:
    print(classifier, 'Mean:', mean, 'std:', std)

# <markdown>
# After experimenting with 11 algorithms, it looks like a neural network type classifier wins the testing round.
#
# Now I need to perform two things:
#     - I will need to delve further into algorithms to understand how they work as to have an idea why they perform either poorly or brilliantly. Algorithms I decided to learn about are:
#         1. MLPClassififer -- to understand neural networks in general
#         2. SGDClassifier -- to understand the stochastic classifier since it's second best
#         3. Decision Tree -- to understand decision trees since it's the worst model to train from the get go.
#     - During testing, I have been wondering about the current dataset and I think I have managed to notice few descrepencies I didn't see before. I will elaborate on this later.
#

# In [ ]
#     ______                  __  _                ___
#    /  _/ /____  _________ _/ /_(_)___  ____     |__ \
#    / // __/ _ \/ ___/ __ `/ __/ / __ \/ __ \    __/ /
#  _/ // /_/  __/ /  / /_/ / /_/ / /_/ / / / /   / __/
# /___/\__/\___/_/   \__,_/\__/_/\____/_/ /_/   /____/
#

# <markdown>
# Purpose of this iteration is to see whether decision trees algorithms will be improved from its worse accuracy performance.

# The plan is to test different set of configurations and then do the same thing for feature engineered features.

# After reading about complex decisions trees that can fail to generalise a problem [Link](https://scikit-learn.org/stable/modules/tree.html#tree),
# the following has been advised to attempt to reduce the chances of such issue:
# - Set a required minimum samples at leaf nodes.
# - Set a depth level number to say how far the tree can go.


# In [ ]
# Testing parameters with original dataset
from sklearn.model_selection import GridSearchCV

decision_tree_clf = DecisionTreeClassifier()
parameters = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_leaf': [0.2, 0.4, 0.5, 1]
}
clf = GridSearchCV(decision_tree_clf, parameters, cv=10)
X = df.loc[:, ['LW','LD','RW','RD']]
y = df.loc[:, ['C']]
clf.fit(X, y)
pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score']).head()

# In [ ]
# Creating columns for the calculated weights.
left_array = df.loc[:, ['LW', 'LD']].to_numpy()
calculations = [item[0] * item[1] for item in left_array]
df['L_calc'] = calculations
right_array = df.loc[:, ['RW', 'RD']].to_numpy()
calculations = [item[0] * item[1] for item in right_array]
df['R_calc'] = calculations
df.head()

# In [ ]
# Testing parameters with a new feature of calculations of weight and height for each side.

decision_tree_clf = DecisionTreeClassifier()
parameters = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_leaf': [0.2, 0.4, 0.5, 1]
}
clf = GridSearchCV(decision_tree_clf, parameters, cv=10)
X = df.loc[:, ['LW','LD','RW','RD', 'L_calc', 'R_calc']]
y = df.loc[:, ['C']]
clf.fit(X, y)
pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score']).head()

# In [ ]
# Test the configurations by using just the calculations of the weights and distance.
decision_tree_clf = DecisionTreeClassifier()
parameters = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_leaf': [0.2, 0.4, 0.5, 1]
}
clf = GridSearchCV(decision_tree_clf, parameters, cv=10)
X = df.loc[:, [ 'L_calc', 'R_calc']]
y = df.loc[:, ['C']]
clf.fit(X, y)
pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score']).head()


# <markdown>
It looks like by introducing the calculations of weights and heights for each side helped the decision tree tremendously. Especially when only providing the calculatins on their own.

I suppose that such calculations is kinda like a cheat, because I think it makes it easier for the model to 'sense' the algorithmic logic.

I am very interested to see what will happen if I introduce boolean flag features (like making a hot-spot (I think) type features that are used for neural networks).

# In [ ]
# Creating feature columns to represent hot-spotted boolean values for the classes.
samples_array = df.loc[:, ['LW', 'LD', 'RW', 'RD']].to_numpy()
df['left_flag'] = [(item[0] * item[1]) > (item[2] * item[3]) for item in samples_array]
df['right_flag'] = [(item[0] * item[1]) < (item[2] * item[3]) for item in samples_array]
df['balanced_flag'] = [(item[0] * item[1]) == (item[2] * item[3]) for item in samples_array]
df.head()

# In [ ]
# Test the configurations by using just the calculations of the weights and distance.
decision_tree_clf = DecisionTreeClassifier()
parameters = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_leaf': [0.2, 0.4, 0.5, 1]
}
clf = GridSearchCV(decision_tree_clf, parameters, cv=10)
X = df.loc[:, ['LW', 'LD', 'RW', 'RD', 'left_flag', 'balanced_flag', 'right_flag']]
y = df.loc[:, ['C']]
clf.fit(X, y)
pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score']).head()

# In [ ]
# Test the configurations by using just the calculations of the weights and distance.
decision_tree_clf = DecisionTreeClassifier()
parameters = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_leaf': [0.2, 0.4, 0.5, 1]
}
clf = GridSearchCV(decision_tree_clf, parameters, cv=10)
X = df.loc[:, ['left_flag', 'balanced_flag', 'right_flag']]
y = df.loc[:, ['C']]
clf.fit(X, y)
pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score']).head()
