# In [ ]
import os
import pickle
from statistics import mean

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
# tree.export_graphviz(clf, out_file="test1.dot",
#                      feature_names=X.columns.to_numpy(),
#                      class_names=pd.unique(y.C.to_numpy()),
#                      filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['LW','LD','RW','RD'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)

# In [ ]
# Introduce engineered weights
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
X = df.loc[:, ['LW','LD','RW','RD','L_calc','R_calc']]
y = df.loc[:, ['C']]
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
clf = DecisionTreeClassifier().fit(X, y)
# tree.export_graphviz(clf, out_file="test2.dot",
#                      feature_names=X.columns.to_numpy(),
#                      class_names=pd.unique(y.C.to_numpy()),
#                      filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['LW','LD','RW','RD','L_calc','R_calc'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)


# In [ ]
# Dataset with only engineered features
X = df.loc[:, ['L_calc','R_calc']]
y = df.loc[:, ['C']]
clf = DecisionTreeClassifier()
scores = cross_val_score(clf, X, y, cv=10)
print('With 10 cross_validation', 'Mean:', scores.mean(), scores.std())
clf = DecisionTreeClassifier().fit(X, y)
# tree.export_graphviz(clf, out_file="test.dot",
#                      feature_names=X.columns.to_numpy(),
#                      class_names=pd.unique(y.C.to_numpy()),
#                      filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['L_calc','R_calc'], clf.feature_importances_)]
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
Updated Observations:
- Ok so for tree with just calcaulted weights, the cross validation showd 97% of the accuracy for the dataset.
    - It's really bizarre that it's roughly 10 percent more accurate than the tree with the original and engineered features despite having very identical tree structure!!! The only thing I deduce about this is that additional probably mess up accuracy of the tree when you give it to predict when a given tree doesn't use this sample. I suppose??? Need to ask somebody about this.

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
# tree.export_graphviz(clf, out_file="weights.dot",
#                      feature_names=X.columns.to_numpy(),
#                      class_names=pd.unique(y.C.to_numpy()),
#                      filled=True, rounded=True)
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
# tree.export_graphviz(clf, out_file="distance.dot",
#                      feature_names=X.columns.to_numpy(),
#                      class_names=pd.unique(y.C.to_numpy()),
#                      filled=True, rounded=True)
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
# tree.export_graphviz(clf, out_file="balanced_without_calc_weights.dot",
#                      feature_names=X.columns.to_numpy(),
#                      class_names=pd.unique(y.C.to_numpy()),
#                      filled=True, rounded=True)
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
# tree.export_graphviz(clf, out_file="balanced_with_calc_weights.dot",
#                      feature_names=X.columns.to_numpy(),
#                      class_names=pd.unique(y.C.to_numpy()),
#                      filled=True, rounded=True)
l = [(name, importance) for name, importance in zip(['LW','LD','RW','RD','L_calc','R_calc'], clf.feature_importances_)]
l.sort(reverse=True, key=lambda i: i[1])
print('Feature importancen')
for name, importance in l:
    print(name, importance)


# <markdown>
Observations about the balanced distribution of the dataset.
- For the balanced dataset without the calculated weights:
    - Interestingly, the tree looked overcomplex. It shows that the tree was overfitting again due to nitpickiness of attributes. It performed poorly as expected (i.e. 53% accuracty with std of 12%).
    - The reason like I said earlier was overfitting and its likely cause I think is overlearning the data to a point of making very intricate rules about each (or handful number of) sample because I think the data attributes couldn't be split further. I need to do some learning though about the algorithm.
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

# <markdown>
# UPDATE: Forget it since the tree can perform very well with the calculated weights alone.
# Make a dataset with a number of left out samples.
# The numbers are 10, 20, 30 for each class in every dataset.
SEED = 1111

for samples_num in [10, 20, 30]:
    # Getting the general
    X = non_sample_df.loc[:, ['LW','LD','RW','RD','L_calc','R_calc']]
    y = non_sample_df.loc[:, ['C']]
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X, y, cv=10)
    print('Cross validation with {0} samples:'.format(samples_num), 'Mean:', scores.mean(), 'Std:', scores.std())

    b_rand_samples = df[df.C == 'B'].sample(n=samples_num, random_state=SEED)
    l_rand_samples = df[df.C == 'L'].sample(n=samples_num, random_state=SEED)
    r_rand_samples = df[df.C == 'R'].sample(n=samples_num, random_state=SEED)
    indices_list = list(b_rand_samples.index.to_numpy()) + list(l_rand_samples.index.to_numpy()) + list(r_rand_samples.index.to_numpy())
    print('Sum of indicies list is', len(indices_list))
    non_sample_indicies = set(df.index.to_numpy()) - set(indices_list)
    print('Sum of non_sample_indicies set is', len(non_sample_indicies))
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

# In [ ]
# Perform to cross_val_score for trees with all features.
from sklearn.model_selection import cross_validate
X = df.loc[:, ['LW','LD','RW','RD','L_calc','R_calc']]
y = df.loc[:, ['C']]
clf = DecisionTreeClassifier()
validations = cross_validate(clf, X, y, cv=10, return_estimator=True, return_train_score=True)
# print('With 10 cross_validation with all features', 'Mean:', scores.mean(), scores.std())
list(validations.keys())
type(validations['estimator'])
print(validations['test_score'])
print(mean(validations['test_score']))



# In [ ]
# Printing balanced tree without engineered features
X = df.loc[:, ['L_calc','R_calc']]
y = df.loc[:, ['C']]
clf = DecisionTreeClassifier()
validations = cross_validate(clf, X, y, cv=10, return_estimator=True, return_train_score=True)
# print('With 10 cross_validation with only engineered features', 'Mean:', scores.mean(), scores.std())
list(validations.keys())
type(validations['estimator'])
print(validations['test_score'])
print(mean(validations['test_score']))

# In [ ]
from sklearn.model_selection import StratifiedKFold

ALL_FEATURES_COLUMNS = ['LW','LD','RW','RD','L_calc','R_calc']
X = df.loc[:, ALL_FEATURES_COLUMNS]
y = df.loc[:, ['C']]
# Creating estimators with all the features
sKf = StratifiedKFold(n_splits=10)
# Format: [{'fitted_estimator', 'mean_score', 'test_split': {'X_test', 'y_test'}, 'train_split': {'X_train', 'y_train'}}]
all_features_cv_list = []
indices_list = []
for train_index, test_index in sKf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    # print(X_train[0:5])
    # print(y_train[0:5])
    # print(X_test[0:5])
    # print(y_test[0:5])
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    mean_score = clf.score(X_test, y_test)
    print('MEAN:', mean_score)
    d = {
        'mean_score': mean_score,
        'fitted_estimator': clf,
        'test_split': {'X_test': X_test, 'y_test': y_test},
        'train_split': {'X_train': X_train, 'y_train': y_train}
    }
    all_features_cv_list.append(d)
    indices_list.append((train_index, test_index))
mean([d['mean_score'] for d in all_features_cv_list])

# It was one hell of a learning experience to realise that for multi-class problems 'StratifiedKFold' is used to split the data.

# Otherwise, I ended up believing that my estimators were 'unicorns' due to their high accuracies that averaged 95% of the time.

# Creating estimators with engineered features only!
ENGINEERED_FEATURES_COLUMNS = ['L_calc','R_calc']
X = df.loc[:, ENGINEERED_FEATURES_COLUMNS]
y = df.loc[:, ['C']]
# Format: [{'fitted_estimator', 'mean_score', 'test_split': {'X_test', 'y_test'}, 'train_split': {'X_train', 'y_train'}}]
engineered_features_cv_list = []
for train_index, test_index in indices_list:
    # print("TRAIN:", train_index, "TEST:", test_index)
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X.to_numpy()[train_index], X.to_numpy()[test_index]
    y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    # print(X_train[0:5])
    # print(y_train[0:5])
    # print(X_test[0:5])
    # print(y_test[0:5])
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    mean_score = clf.score(X_test, y_test)
    print('MEAN:', mean_score)
    d = {
        'mean_score': mean_score,
        'fitted_estimator': clf,
        'test_split': {'X_test': X_test, 'y_test': y_test},
        'train_split': {'X_train': X_train, 'y_train': y_train}
    }
    engineered_features_cv_list.append(d)
mean([d['mean_score'] for d in engineered_features_cv_list])

# Function for producing dot and pdfs files
def make_pdf(name, index, clf, columns, uniq_names):
    TEMPLATE = '{name}_{index}'
    dot_name = TEMPLATE.format(name=name, index=index) + ".dot"
    tree.export_graphviz(clf, out_file='produced_pdfs/'+dot_name,
                         # feature_names=X.columns.to_numpy(),
                         feature_names=columns,
                         class_names=uniq_names,
                         filled=True, rounded=True)
    os.system('dot -Tps produced_pdfs/{0}.dot -o produced_pdfs/{0}.pdf'.format(TEMPLATE.format(name=name, index=index)))

# Calculating performance differences between trained classifiers
UNIQUE_CLASS_NAMES = pd.unique(y.C.to_numpy())

performance_diff_list = []
engineered_perf_list = []
all_f_perf_list = []
counter_list = []
for index, dict_tuple in enumerate(zip(all_features_cv_list, engineered_features_cv_list)):
    all_f_dict, engineered_f_dict = dict_tuple
    # It's expected for engineered features to perform better thus substracting from its performance score
    performance_diff_list.append(engineered_f_dict['mean_score'] - all_f_dict['mean_score'])
    engineered_perf_list.append(engineered_f_dict['mean_score'])
    all_f_perf_list.append(all_f_dict['mean_score'])
    counter_list.append(index)
    # Creating pdfs!
    # make_pdf('all_f', index, all_f_dict['fitted_estimator'], ALL_FEATURES_COLUMNS, UNIQUE_CLASS_NAMES)
    # make_pdf('engi_f', index, engineered_f_dict['fitted_estimator'], ENGINEERED_FEATURES_COLUMNS, UNIQUE_CLASS_NAMES)

# os.system('rm produced_pdfs/*.dot')

data = pd.DataFrame({
    'cv_iter': pd.Series(counter_list),
    'engineered_perf': pd.Series(engineered_perf_list),
    'all_perf': pd.Series(all_f_perf_list),
    'perf_diff': pd.Series(performance_diff_list),
})

data

'''
# Results at the time of generating the files.
# If you run this notebook/file the results may vary slightly.
cv_iter	engineered_perf	all_perf	perf_diff
0	0	0.936508	0.888889	0.047619
1	1	1.000000	0.968254	0.031746
2	2	0.968254	0.936508	0.031746
3	3	1.000000	1.000000	0.000000
4	4	1.000000	0.873016	0.126984
5	5	0.983871	0.758065	0.225806
6	6	1.000000	1.000000	0.000000
7	7	0.983871	0.693548	0.290323
8	8	0.967742	0.870968	0.096774
9	9	0.935484	0.854839	0.080645
'''

# In [ ]
def save_stuff(all_features_cv_list, engineered_features_cv_list, data):
    # (Hopefully) saving the current state of the classifiers and test and train indicies to a file
    with open('saves/all_features_cv_list.pickle', 'wb') as file_obj:
        pickle.dump(all_features_cv_list, file_obj)
    with open('saves/engineered_features_cv_list.pickle', 'wb') as file_obj:
        pickle.dump(engineered_features_cv_list, file_obj)
    data.to_pickle('saves/data_dataframe.pickle')

def get_stuff():
    with open('saves/all_features_cv_list.pickle', 'rb') as file_obj:
        resurectted = pickle.load(file_obj)
    with open('saves/engineered_features_cv_list.pickle', 'rb') as file_obj:
        resurectted_2 = pickle.load(file_obj)
    resurect_data = pd.read_pickle('saves/data_dataframe.pickle')
    return resurectted, resurectted_2, resurect_data

# In [ ]
# Getting stuff back from saves files
all_features_cv_list, engineered_features_cv_list, data = get_stuff()
data

# In [ ]
from collections import Counter
def display_info(index, all_f_dict, engineered_f_dict):
    '''Function to display informative values about a CV iteration.
    TO-DO:
    - Displaying a list of failed samples for each classifier
    '''
    print('Iteration_CV:', index, 'All_features_score:', all_f_dict['mean_score'], 'Engineered_features_score:', engineered_f_dict['mean_score'], 'Perf_diff:', engineered_f_dict['mean_score'] - all_f_dict['mean_score'])

    # Since the test and training data splits are the same for engineered and features
    print('Training_data_total', len(all_f_dict['train_split']['X_train']), 'Testing_data_total', len(all_f_dict['test_split']['X_test']))
    # But it won't escape checking ;)
    # 4, 5 indicies for the all features since that's the calcualted weights.
    print([[sample[4], sample[5]] for sample in all_f_dict['train_split']['X_train']] == engineered_f_dict['train_split']['X_train'])
    all_f_samples = [[sample[4], sample[5]] for sample in all_f_dict['train_split']['X_train']]
    assert all([all(l) for l in all_f_samples == engineered_f_dict['train_split']['X_train']])
    assert all([all(l) for l in all_f_dict['train_split']['y_train'] == engineered_f_dict['train_split']['y_train']])
    all_f_samples = [[sample[4], sample[5]] for sample in all_f_dict['test_split']['X_test']]
    assert all([all(l) for l in all_f_samples == engineered_f_dict['test_split']['X_test']])
    assert all([all(l) for l in all_f_dict['test_split']['y_test'] == engineered_f_dict['test_split']['y_test']])

    # Printing percentages of classes
    # print('hello there', [i[0] for i in all_f_dict['train_split']['y_train']])
    train_classes_counter = Counter([i[0] for i in all_f_dict['train_split']['y_train']])
    test_classes_counter = Counter([i[0] for i in all_f_dict['test_split']['y_test']])
    total_train_samples = len(all_f_dict['train_split']['y_train'])
    total_test_samples = len(all_f_dict['test_split']['y_test'])
    train_sample_percentages = [(sample_class, (count/total_train_samples)*100) for sample_class, count in train_classes_counter.most_common()]
    test_sample_percentages = [(sample_class, (count/total_test_samples)*100) for sample_class, count in test_classes_counter.most_common()]
    print('train_samples_classes_percentages:', train_sample_percentages)
    print('test_samples_classes_percentages:', test_sample_percentages)

    # Print successful and failed samples for all features classifier
    test_split_X = all_f_dict['test_split']['X_test']
    test_split_y = all_f_dict['test_split']['y_test']
    all_f_clf = all_f_dict['fitted_estimator']
    predicted_classes = all_f_clf.predict(test_split_X)
    print('predicted_classes:', predicted_classes)

    # Getting separate attributes for each sample for all features
    features_dict = {}
    for index, feature in enumerate(ALL_FEATURES_COLUMNS):
        features_dict[feature] = [a[index] for a in test_split_X]

    all_f_dict = {
        'returned_class': pd.Series(predicted_classes),
        'expected_class': pd.Series([i[0] for i in test_split_y])
    }
    all_f_dict = {**all_f_dict, **features_dict}
    all_f_df = pd.DataFrame(all_f_dict)
    matches = [row['returned_class'] == row['expected_class'] for index, row in all_f_df.iterrows()]
    all_f_df['matched'] = pd.Series(matches)
    print(all_f_df.groupby(['expected_class', 'matched']).count())


# In [ ]
display_info(9, all_features_cv_list[9], engineered_features_cv_list[9])

ad = all_features_cv_list[9]
ed = engineered_features_cv_list[9]

print('ad:', ad['train_split']['X_train'][:25])
print('ed:', ed['train_split']['X_train'][:25])
