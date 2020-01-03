import pandas as pd
import seaborn as sns

sns.set()

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
# 1. Need to run some models to see that the proportionality of the will make a very accurate model.
# 2. Need to find a way to show percentages of classes for each intersecting point
# to see whether there could be noise for the classes.
