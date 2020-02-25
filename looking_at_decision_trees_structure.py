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
The purpose of this file is to present (and generate) decision tree structures
as to see why they are as effective with and without engineered features.

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
Observations about the produced pdfs.
- It looks like that tree with the calculated weights has a lot less horizontal spread compared to one without them.
    - I believe that's reason the decision tree with the weights performs a lot better because it captured more **generality** thus not making the structure bigger.
        - Algorithm Knowledge Update: The reason it managed to capture the mentioned "generality" is because the 'Gini' impurity calculation showed that the new features benefitted the tree due to samples being split better (i.e. a set of samples where most of them lean toward a particular class rather than having an almost equal number of them leaning towards three).
    - Also, this definitely shows that the tree is not overfitted. This is because a symptoms of overfitting is when a tree is over-complex like the one without the calculated weights.
- The tree with the weights prefers calculated weights over original features.
    - This is probably because fewer features (i.e. the calculated weights) were more useful for generalisation rather than specific measures of the scales. The documentation mentioned that data with a large amount of features tend to overfit. Even though I only got four original features, the algorithm prefered the weight calculations nontheless. Need to generate trees with just a distance and weights and see how they develop.
- Generating two tree structures that only used either weights or distances:
    - After generating the trees, they looked identical to each other. I believe it's because the sample values were the same for both measurements that the tree just looked identical. Also performance was not worse or better: 60% accuracy for distance and 65% for weights.
        - Algorithm Knowledge Update: I believe it's because of the huge amount of samples for 'left' and 'right' classes that the tree was very biased for them. This is because the 'Gini' impurity calculation proved that it couldn't go any lower due to being the 'best' split for samples. Thus, the leafs would be biased for two classes even though there were some 'Balanced' samples present in them.
    - I noticed that the trees were very biased: There were no 'Balanced' class present at the leafs of the tree. It looks like the tree was not 'growing' completely without leafs representing the 'Balanced' class.
    - Also, noticed that the trees had very weird leafs where it clould not decide whether it should have been 'Left' or 'Right' classes.
        - Algorithm Knowledge Update: I believe the reason is that it couldn't split a given sample set at that point since its impurity was already low enough compared to calculated splits' ones. Thus, it produced a leaf where it just lingers in limbo of deciding whether it was right or left balanced.


# In [ ]
# Dataset with engineered features
X = df.loc[:, ['LW','RW']]
y = df.loc[:, ['C']]
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
clf = DecisionTreeClassifier().fit(X, y)
tree.export_graphviz(clf, out_file="weights.dot",
                     feature_names=X.columns.to_numpy(),
                     class_names=pd.unique(y.C.to_numpy()),
                     filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['LW','LD'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)


# In [ ]
# Dataset with engineered features
X = df.loc[:, ['RD','LD']]
y = df.loc[:, ['C']]
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
clf = DecisionTreeClassifier().fit(X, y)
tree.export_graphviz(clf, out_file="distance.dot",
                     feature_names=X.columns.to_numpy(),
                     class_names=pd.unique(y.C.to_numpy()),
                     filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['RW','RD'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)

# In [ ]
# Dataset with balanced number of classes
TOTAL_NUMBER = df[df.C == 'B'].C.count()
balanced = df[df.C == 'B']
print(TOTAL_NUMBER)
right_side = df[df.C == 'R'].sample(n=TOTAL_NUMBER)
print(right_side.count())
left_side = df[df.C == 'L'].sample(n=TOTAL_NUMBER)
print(left_side.count())
balanced_samples_df = pd.concat([balanced, left_side, right_side])
balanced_samples_df.describe()

# In [ ]
# Printing balanced tree without engineered features
X = balanced_samples_df.loc[:, ['LW','LD','RW','RD']]
y = balanced_samples_df.loc[:, ['C']]
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
clf = DecisionTreeClassifier().fit(X, y)
tree.export_graphviz(clf, out_file="balanced_without_calc_weights.dot",
                     feature_names=X.columns.to_numpy(),
                     class_names=pd.unique(y.C.to_numpy()),
                     filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['LW','LD','RW','RD'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)

# In [ ]
# Printing balanced tree without engineered features
X = balanced_samples_df.loc[:, ['LW','LD','RW','RD','L_calc','R_calc']]
y = balanced_samples_df.loc[:, ['C']]
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
clf = DecisionTreeClassifier().fit(X, y)
tree.export_graphviz(clf, out_file="balanced_with_calc_weights.dot",
                     feature_names=X.columns.to_numpy(),
                     class_names=pd.unique(y.C.to_numpy()),
                     filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['LW','LD','RW','RD','L_calc','R_calc'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)


# <markdown>
Observations about the balanced distribution of the dataset.
- For the balanced dataset without the calculated weights:
    - Interestingly, the tree looked overcomplex. It shows that the tree was overfitting again due to nitpickiness of attributes. It performed poorly as expected (i.e. 53% accuracty with std of 12%).
    - The reason like I said earlier was overfitting and its likely cause I think is overlearning the data to a point of making very intricate rules about each data because I think the data attributes couldn't be split further. I need to do some learning though about the algorithm.
- For the balanced dataset without the calculated weights.
    - It still looks kinda complex but it's not as big or wide as the previous tree. This is because the tree preferred the calculated weights over the original features and thus it managed to generalise the problem (i.e. the accuracy score is 65% with 14% standard deviation).
    - Even though the structure looks kinda balanced on both sides, there's a subtree at the bottom where it goes 4 branches deep. I assume it is because there were some samples left at a point where it seemed it had converged that it was very difficult to equally split the samples based on a best attribute. Thus creating a tiny subtree.

# <markdown>
To-Do for figuring the inaccuracy of the model:
1. Build a model with most of samples for all classes but leave out some samples of all classes for testing purposes:
    - Check how many balanced have failed and passed.
    - Check how many right side classses failed and passed.
    - Check how many left side classses failed and passed.
2. Build a model with balanced dataset but leave out some samples for the balanced class and the rest of the samples:
    - Check how many balanced have failed and passed.
    - Check how many right side classses failed and passed.
    - Check how many left side classses failed and passed.

# In [ ]
# Make a dataset with a number of left out samples.
# The numbers are 10, 20, 30 for each class in every dataset.
SEED = 1111

for samples_num in [10, 20, 30]:
    b_rand_samples = df[df.C == 'B'].sample(n=samples_num, random_state=SEED)
    l_rand_samples = df[df.C == 'L'].sample(n=samples_num, random_state=SEED)
    r_rand_samples = df[df.C == 'R'].sample(n=samples_num, random_state=SEED)
    indices_list = b_rand_samples.index.to_numpy() + l_rand_samples.index.to_numpy() + r_rand_samples.index.to_numpy()
    print('Sum of indicies list is', len(indices_list))
    non_sample_indicies = set(df.index.to_numpy()) - set(indices_list)
    print('Sum of non_sample_indicies set is', len(non_sample_indicies))
    non_sample_df = df.iloc[sort(list(non_sample_indicies))]
    print("Sorted Non sample df head:\n", non_sample_df.head())
    non_sample_df = df.iloc[list(non_sample_indicies)]
    print("Non sample df head:\n", non_sample_df.head())

    X = non_sample_df.loc[:, ['LW','LD','RW','RD','L_calc','R_calc']]
    y = non_sample_df.loc[:, ['C']]
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X, y, cv=10)
    print('Cross validation with {0} samples:'.format(samples_num), 'Mean:', scores.mean(), 'Std:', scores.std())
    clf = DecisionTreeClassifier().fit(X, y)
    # Need to write cross scores
    # Uncomment once the code is ready
    # tree.export_graphviz(clf, out_file="balanced_with_calc_weights.dot",
    #                      feature_names=X.columns.to_numpy(),
    #                      class_names=pd.unique(y.C.to_numpy()),
    #                      filled=True, rounded=True)
