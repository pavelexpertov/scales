import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

# The dataset is from https://archive.ics.uci.edu/ml/datasets/Balance+Scale

# In [ ]
d = pd.read_csv('balance-scale.data', header=None, names=['C','LW','LD','RW','RD'])
d

# <markdown>
# Looks like total number of examples is here.

# In [ ]
d.describe()

# In [ ]
d.C.describe()
# <markdown>
# It looks like there's no discrepencies with values
# (i.e. all values seem to be in range of 1 to 5 inclusive).
#
# Let's draw some graphs.

# In [ ]
sns.relplot(x="LD", y="RD", hue="C", data=d)


# In [ ]
sns.relplot(x="LD", y="RD", hue="C", data=d)


# In [ ]
sns.relplot(x="LW", y="RW", hue="C", data=d)

# In [ ]
sns.relplot(x="LW", y="RW", hue="C", data=d)

# <markdown>
# By looking at these graphs, it looks like values are proportianally distributed
# whether it's against the same attributes or not.
#
# Will need to do two things:
# 1. Need to run some models to see that the proportionality of the height and weight will make a very accurate model.
# 2. Need to find a way to show percentages of classes for each intersecting point (no point -- the classes do not have duplicate samples. Thus no noise.)
# to see whether there could be noise for the classes.


# <markdown>
# After running some models and concluding results about their accuricies. I wondered what kind of data I am dealing with.

# In [ ]
d[d.C == 'B'].describe()
# In [ ]
samples = d[d.C == 'B'].loc[:, ['LW', 'RW', 'LD', 'RD']].to_numpy()
s = set([tuple(array) for array in samples])
print('Sum of items:', len(s))
# Looks like there are unique values for the Balanced class

# In [ ]
d[d.C == 'L'].describe()

# In [ ]
samples = d[d.C == 'L'].loc[:, ['LW', 'RW', 'LD', 'RD']].to_numpy()
s = set([tuple(array) for array in samples])
print('Sum of items:', len(s))
# Looks like there are unique values for the Left class.

# In [ ]
d[d.C == 'R'].describe()

# In [ ]
samples = d[d.C == 'R'].loc[:, ['LW', 'RW', 'LD', 'RD']].to_numpy()
s = set([tuple(array) for array in samples])
print('Sum of items:', len(s))
# Looks like there are unique values for the Right class.

# In [ ]
# Let's see whether different classes share the same samples
samples = d[d.C == 'B'].loc[:, ['LW', 'RW', 'LD', 'RD']].to_numpy()
B_SET = set([tuple(array) for array in samples])

samples = d[d.C == 'L'].loc[:, ['LW', 'RW', 'LD', 'RD']].to_numpy()
L_SET = set([tuple(array) for array in samples])

samples = d[d.C == 'R'].loc[:, ['LW', 'RW', 'LD', 'RD']].to_numpy()
R_SET = set([tuple(array) for array in samples])

# In [ ]
# Intersection between left and right class
print(len(L_SET & R_SET))

# In [ ]
# Intersection between left and balanced class
print(len(L_SET & B_SET))

# In [ ]
# Intersection between balanced and right class
print(len(B_SET & R_SET))

# In [ ]
# In the end, I think I could have just done this:
d.duplicated(subset=['LW', 'RW', 'LD', 'RD']).describe()
# The unique value is one, the top value is false and it has all the entries. Thus, no duplicates.

# <markdown>
After performing certain checks, I have the following thoughts:
- The dataset doesn't have any noise since there are no duplicate samples that classes share with each other.
- The range of values that exist for the balanced class indicates that the dataset represents multiple scales rather than one.

The last point really make sense to me after rereading the dataset description.

The description said that the dataset was generated to 'model psychological experiments' and there is a simple logic algorithm to determine which class of the balance is
based on readings of weight and distance for each side.

It means that the nature of dataset was to test algorithm's ability to train a model that would 'understand' or 'perceive' the logic for concluding classes correctly.

It really explains why neural networks had the accuracy of 97% with low standard deviation since they learnt on mistakes and generalise the logic problem compared to
other models.

# In [ ]
#     ______                  __  _                ___
#    /  _/ /____  _________ _/ /_(_)___  ____     |__ \
#    / // __/ _ \/ ___/ __ `/ __/ / __ \/ __ \    __/ /
#  _/ // /_/  __/ /  / /_/ / /_/ / /_/ / / / /   / __/
# /___/\__/\___/_/   \__,_/\__/_/\____/_/ /_/   /____/
#

# <markdown>
After learning about the decision tree algorithms, I will try to engineer features that can help the algorithm to work better.

The following comes to mind:
1. Make two features to represent calculations for each side.
2. Make three features to represent boolean values of left, right or balanced calculations. (i.e. make it look like a binary pattern like 1,0,0 for tipping to the left)
3. Make a balanced dataset (i.e. Make an equal amount of examples for right and left classes since there are way more of them rather than balanced one).
    1. Split it with all examples of the balanced class.
    2. Split it with half examples of the balanced class.
    3. Split it with one examples of the balanced class.
