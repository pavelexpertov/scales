# In [ ]
import os
import pickle
from statistics import mean
from icecream import ic

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
    print('Iteration_CV:', index)
    print('All_features_score:', all_f_dict['mean_score'])
    print('Engineered_features_score:', engineered_f_dict['mean_score'])
    print('Perf_diff:', engineered_f_dict['mean_score'] - all_f_dict['mean_score'])

    # Since the test and training data splits are the same for engineered and features
    print('Training_data_total', len(all_f_dict['train_split']['X_train']), 'Testing_data_total', len(all_f_dict['test_split']['X_test']))
    # But it won't escape checking ;)
    # 4, 5 indicies for the all features since that's the calcualted weights.
    all_f_samples = [[sample[4], sample[5]] for sample in all_f_dict['train_split']['X_train']]
    assert all([all(l) for l in all_f_samples == engineered_f_dict['train_split']['X_train']])
    assert all([all(l) for l in all_f_dict['train_split']['y_train'] == engineered_f_dict['train_split']['y_train']])
    all_f_samples = [[sample[4], sample[5]] for sample in all_f_dict['test_split']['X_test']]
    assert all([all(l) for l in all_f_samples == engineered_f_dict['test_split']['X_test']])
    assert all([all(l) for l in all_f_dict['test_split']['y_test'] == engineered_f_dict['test_split']['y_test']])

    # Printing percentages of classes
    train_classes_counter = Counter([i[0] for i in all_f_dict['train_split']['y_train']])
    test_classes_counter = Counter([i[0] for i in all_f_dict['test_split']['y_test']])
    total_train_samples = len(all_f_dict['train_split']['y_train'])
    total_test_samples = len(all_f_dict['test_split']['y_test'])
    train_sample_percentages = [(sample_class, round((count/total_train_samples)*100, 2)) for sample_class, count in train_classes_counter.most_common()]
    test_sample_percentages = [(sample_class, round((count/total_test_samples)*100, 2)) for sample_class, count in test_classes_counter.most_common()]
    print('train_samples_classes_percentages:', train_sample_percentages)
    print('test_samples_classes_percentages:', test_sample_percentages)

    # Print successful and failed samples for all features classifier
    test_split_X = all_f_dict['test_split']['X_test']
    test_split_y = all_f_dict['test_split']['y_test']
    all_f_clf = all_f_dict['fitted_estimator']
    predicted_classes = all_f_clf.predict(test_split_X)
    # print('all_f_predicted_classes:', predicted_classes)

    # Getting separate attributes for each sample for all features
    features_dict = {}
    for index, feature in enumerate(ALL_FEATURES_COLUMNS):
        features_dict[feature] = [a[index] for a in test_split_X]

    all_f_for_df_dict = {
        'returned_class': pd.Series(predicted_classes),
        'expected_class': pd.Series([i[0] for i in test_split_y])
    }
    all_f_for_df_dict = {**all_f_for_df_dict, **features_dict}
    all_f_df = pd.DataFrame(all_f_for_df_dict)
    matches = [row['returned_class'] == row['expected_class'] for index, row in all_f_df.iterrows()]
    all_f_df['matched'] = pd.Series(matches)
    print('All Features table')
    print(all_f_df.groupby(['expected_class', 'matched']).count())

    # Print successful and failed samples for engineered features classifier
    test_split_X = engineered_f_dict['test_split']['X_test']
    test_split_y = engineered_f_dict['test_split']['y_test']
    engineered_f_clf = engineered_f_dict['fitted_estimator']
    predicted_classes = engineered_f_clf.predict(test_split_X)
    # print('engineered_predicted_classes:', predicted_classes)

    # Getting separate attributes for each sample for all features
    features_dict = {}
    for index, feature in enumerate(ENGINEERED_FEATURES_COLUMNS):
        features_dict[feature] = [a[index] for a in test_split_X]

    engineered_f_for_df_dict = {
        'returned_class': pd.Series(predicted_classes),
        'expected_class': pd.Series([i[0] for i in test_split_y])
    }
    engineered_f_for_df_dict = {**engineered_f_for_df_dict, **features_dict}
    engineered_f_df = pd.DataFrame(engineered_f_for_df_dict)
    matches = [row['returned_class'] == row['expected_class'] for index, row in engineered_f_df.iterrows()]
    engineered_f_df['matched'] = pd.Series(matches)
    print('Engineered Features table')
    print(engineered_f_df.groupby(['expected_class', 'matched']).count())

    # Create a dataframe where samples from all_f_df failed as well as succeed in the engineered_f_df
    row_index_list = []
    for index, tuples in enumerate(zip(all_f_df.itertuples(), engineered_f_df.itertuples())):
        all_f_row, engineered_f_row = tuples[0], tuples[1]
        if not all_f_row.matched and engineered_f_row.matched:
            row_index_list.append(index)

    valid_samples_df = all_f_df.iloc[row_index_list]
    print("Counts")
    print(valid_samples_df.count())
    return valid_samples_df

def list_decision_travel_nodes(all_f_dict, LW, LD, RW, RD, L_calc, R_calc):
    '''List nodes where a sample traversed through whilst predicted a sample'''
    # Print successful and failed samples for all features classifier
    classifier = all_f_dict['fitted_estimator']
    X_test = np.ndarray([LW, LD, RW, RD, L_calc, R_calc])
    X_test = [X_test]

    ic(dir(classifier))
    ic(classifier.n_features_)
    node_indicator = classifier.decision_path(X_test)
    leave_id = classifier.apply(X_test)

    sample_id = 0 # Since I only provided one sample, the sample id should be 0 by default
    node_index = node_indicator.indicies[node_indicator.indptr[sample_id],
                                         node_indicator.indptr[sample_id + 1]]

    print("Decision path for the provided paramters:")
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 X_test[sample_id, feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))
    # Unfortunately will have to stop implementing the path printer since I encountered an error that can't problem solve.
    # Going to explore the tree manually to see why 7th cross validation has got majority of samples failing


# In [ ]
# Trees
display_info(7, all_features_cv_list[7], engineered_features_cv_list[7])

# In [ ]
# Trees that have zero difference in performance
display_info(3, all_features_cv_list[3], engineered_features_cv_list[3])
display_info(6, all_features_cv_list[6], engineered_features_cv_list[6])

# <markdown>
Ok, so this is not suprising since the trees (almost) closely identically to one another so there's zero difference in performance.

3 and 3 iterations all_f: the tree structures look very identical. Except for one leftist node. One tree has used L_calc whereas the other used R_calc. I believe this is because at that node, the attributes would have the same Gini value and thus it would've randomly selected the attribute.
Despite that, leaves are the same between trees.

6 and 6 iterations engineered_f:
Trees look very idnetical.


# In [ ]
# Trees with roughly 10 performance difference on average
# 4, 8, 9
display_info(4, all_features_cv_list[4], engineered_features_cv_list[4])
display_info(8, all_features_cv_list[8], engineered_features_cv_list[8])
display_info(9, all_features_cv_list[9], engineered_features_cv_list[9])

# <markdown>
Iteration 4:
    - It looks like the trees look very identical, but the all_f tree had used an original attribute right in the centre of the tree. Possibly because of it the all_f tree under performed. However, need to implement the diagnostic function to see where the path travel.

Iteration 8:
    - Trees look almost identical, but again, an original feature introduced an additional depth to the all_f tree compared to engineered_f one.

Iteration 9:
    - Trees look identical, but all_f's three nodes use original features to split the samples. However, it does look like leaves are the same between the trees.

In conclusion for the specified iterations, the trees almost look alike, but the original features either alter a structure of the tree a bit or a node, which would usually use an engineered node.

My assumption is that the reason the original features are used is because at that particular node had either the same amount of samples that the splits were so similar that the attribute was chosen at random *or* the split was a bit better than the engineeered ones.

And that's why there were a handful of mismatches for each class in iterations since a particular set of samples would have gone a path that wasn't generalised/trained on it properly.

# In [ ]
# Trees with roughly 3 to 4 percent different in performance
# 0, 1, 2
display_info(0, all_features_cv_list[0], engineered_features_cv_list[0])
display_info(1, all_features_cv_list[1], engineered_features_cv_list[1])
display_info(2, all_features_cv_list[2], engineered_features_cv_list[2])

# In [ ]
# Trees with bigger performance differences.
# 5, 7
display_info(5, all_features_cv_list[5], engineered_features_cv_list[5])
display_info(7, all_features_cv_list[7], engineered_features_cv_list[7])

# <markdown>
# Trees with the worst performance
Iteration 5
    - The trees look very familiar but there's one path that stands out. In the all features tree, on a second node ('L_calc' <= 13.5), a 'LD' feature was selected and as a result the structure of the path looks different to the path in the engineered tree: the feature gave itself an extra two depths and it had some leaves along the way whereas the engineered tree had a balanced subtree.
        - Ok this is weird: for the right most node (in all features tree), the gini confidence is the same as that of the engineered branch. I think the value for the gini confidence was still different because the values are rounded and therefore the values would be very close to between each other but still different. (will need to math for that one)
        - Also, by looking at the failed samples (which engineered_f tree succeeded), it looks like for samples that have only 2 for 'LW' and 5 'LD' features made the trained model fail by making it think that it's balanced even if the other 4 features indicated that it was leaning on the right.
            - Note that it's the 10 calculated weight.
            - By looking at the tree, I followed that the samples that got the class wrong would have to go through (LD <= 4.5) condition on right most path of the tree from the second node of (L_calc <= 13.5).
        - For other failed samples,
            - samples that had balanced labels had a thing where each two original feature of each side were the same and either in a repeating or mirrored pattern (look at the table with index of 30, 31). My guess is that a particular node that would get one of the sample would lead them in wrong a path because the mathematical rule dimmed it 'best' to split the dataset at that time.
            - Similar issue hapens for the 38 and 45 index where it exted left leaning side rather than balanced.
Iteration 7
    - What's interesting is that half of the left side classes samples fail completely in comparison to the right side (even though the balanced class has 50/50 success/failire rate as well).
    - The trees again look identiacal, but the all_f tree has a node that disticly altered the shape of a subtree, especially for the node under the second node on hte right side (i.e. R_calc <= 13.5). The subtree uses an original attribute to split the samples and it introduces an additional node that would seperate an a subtree strucutre very similar to the engineered_f subtree under the same node I mentioned.
    - There's another node that uses the the original attribute but the leaves under it seem the same between the trees.
    - Looking at the produced table of failed samples:
        - It looks like that samples (from 40 to 54 indicies) share three features in common: LW=5, LD=2 and L_calc=10. These samples expected a left class but got balanced class instead.
            - Ok, by looking at the tree and the node that I noticed that looked out of place (i.e. R_calc <= 13.5) and its following nodes of (LD <= 2.5) and (R_calc <=11.0) will make the samples mentioned into the balanced label even if it's incorrectly mathematically. My speculation is that the tree got overtrained on a particular set of samples that made the subtree with aforementioned nodes.

In conclusion, it seems original attributes, which have been selected as the best sample splitting attribute according to Gini value, affect the structure of the trees in such a way that tested sample fail the most. Therefore, I need to confirm that these attributes affect the classification prediciton of these samples by looking at the path that attributes take.

*Additional* conclusion: I noticed something really weird between two trees: it's not only the original feature that ruined the subree creation as to create some sort of overfitting, it's the node that were produced by such feature had either a node or a subtree with 1 or 2 depths that had very few samples at the leaves. In other words, even though the tree was trained to accommodate these few examples, it made the tree fail at getting that 'generalisation' it needs to be accurate... ehm see distinction between two classes.
