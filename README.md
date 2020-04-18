# Scales -- First Machine Learning project for practice

The dataset here is about measurements of distance and weight for each side of the scale. [Original source](https://archive.ics.uci.edu/ml/datasets/Balance+Scale) for the dataset.

A written blog post about this work is [here](need the source).

The task is to create a predictive model that essentially predicts whether the scales is in a state of having either side (i.e. left or right) tipped up or in balance.

The jupyter notebooks to explore what work has been done to achieve this:
- [looking_at_data.ipynb](https://github.com/pavelexpertov/scales-ml/blob/master/looking_at_data.ipynb) -- explore the dataset of weight and distance measurements.
- [training_models.ipynb](https://github.com/pavelexpertov/scales-ml/blob/master/training_models.ipynb) -- exploring different algorithms.
- [looking_at_decision_trees_structure.ipynb](https://github.com/pavelexpertov/scales-ml/blob/master/looking_at_decision_trees_structure.ipynb) -- exploring factors that affect decision tree's accuracy and observe its trained models and their structures.

Even though there is no code to 'interactively' play with trained models, it is possible to re-produce the environment the work is completed in. If you have `conda` command installed, then...
```shell
conda env create -f environment.yml
conda activate scales2
jupyter notebook
```
