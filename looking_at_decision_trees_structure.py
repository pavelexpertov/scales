# In [ ]
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

sns.set()

# The dataset is from https://archive.ics.uci.edu/ml/datasets/Balance+Scale

# In [ ]
df = pd.read_csv('balance-scale.data', header=None,
                names=['C','LW','LD','RW','RD'])
df

# <markdown>
# The purpose of this file is to present (and generate) decision tree structures
# as to see why they are as effective with and without engineered features.

# In [ ]
# Preparing dataset for training and testing under normal conditions

X = df.loc[:, ['LW','LD','RW','RD']]
y = df.loc[:, ['C']]
X.columns.to_numpy()
pd.unique(y.C.to_numpy())
# In [ ]
# Normal dataset
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
clf = DecisionTreeClassifier()
clf.fit(X, y)
tree.export_graphviz(clf, out_file="test1.dot",
                     feature_names=X.columns.to_numpy(),
                     class_names=pd.unique(y.C.to_numpy()),
                     filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['LW','LD','RW','RD'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)

# <markdown>
# With 10 cross_validation Mean: 0.6737839221710189 0.10366816319107584
# Feature importancen
# RW 0.26950180674161495
# LD 0.26188612263854244
# LW 0.24044564183138226
# RD 0.22816642878846036


# In [ ]
# Introduce weights
left_array = df.loc[:, ['LW', 'LD']].to_numpy()
calculations = [item[0] * item[1] for item in left_array]
df['L_calc'] = calculations
right_array = df.loc[:, ['RW', 'RD']].to_numpy()
calculations = [item[0] * item[1] for item in right_array]
df['R_calc'] = calculations
X = df.loc[:, ['LW','LD','RW','RD','L_calc','R_calc']]
y = df.loc[:, ['C']]

# In [ ]
# Dataset with engineered features
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
clf = DecisionTreeClassifier().fit(X, y)
tree.export_graphviz(clf, out_file="test2.dot",
                     feature_names=X.columns.to_numpy(),
                     class_names=pd.unique(y.C.to_numpy()),
                     filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['LW','LD','RW','RD','L_calc','R_calc'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)

# <markdown>
# Values of feature importances and the score:
# With 10 cross_validation Mean: 0.8860727086533539 0.09875318928663263
# Feature importancen
# L_calc 0.5175521558099616
# R_calc 0.48244784419003833
# LW 0.0
# LD 0.0
# RW 0.0
# RD 0.0

# <markdown>
# Observations about the produced pdfs.
- It looks like that tree with the calculated weights has a lot less horizontal spread compared to one without them.
    - I believe that's reason the decision tree with the weights performs a lot better because it captured more **generality** thus not making the structure bigger.
- The tree with the weights prefers calculated weights over original features.
    - This is probably because
