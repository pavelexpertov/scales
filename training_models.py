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
After experimenting with 11 algorithms, it looks like a neural network type classifier wins the testing round.

Now I need to perform two things:
    - I will need to delve further into algorithms to understand how they work as to have an idea why they perform either poorly or brilliantly. Algorithms I decided to learn about are:
        1. MLPClassififer -- to understand neural networks in general
        2. SGDClassifier -- to understand the stochastic classifier since it's second best
        3. Decision Tree -- to understand decision trees since it's the worst model to train from the get go.
    - During testing, I have been wondering about the current dataset and I think I have managed to notice few descrepencies I didn't see before. I will elaborate on this later.
