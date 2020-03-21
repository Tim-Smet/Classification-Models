"""
Predicting student performance with decision tree. Pass or Fail
"""
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset (student portugese scores)
dataset = pd.read_csv('student-por.csv', sep=';')

#We will add two columns for pass and fail.
#This will be done with the apply function of pandas. if G1 + G2 + G3 >= 35 it is pass.
dataset['pass'] = dataset.apply(lambda row: 1 if (row['G1'] + row['G2'] + row['G3']) >= 35 else 0, axis=1)
dataset = dataset.drop(['G1', 'G2', 'G3'], axis=1)

#We have columns with words or phrases. Need to be converted to numbers with pandas function get_dummies
#One-hot encoding
#We will end up with 57 columns
dataset = pd.get_dummies(dataset, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup',
                                           'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])

#We will shuffle the rows and devide them in a test set (149) and training set (500)
dataset = dataset.sample(frac=1)
dataset_train = dataset[:500]
dataset_test = dataset[500:]

dataset_train_att = dataset_train.drop(['pass'], axis=1)
dataset_train_pass = dataset_train['pass']
dataset_test_att = dataset_test.drop(['pass'], axis=1)
dataset_test_pass = dataset_test['pass']
dataset_att = dataset.drop(['pass'], axis=1)
dataset_pass = dataset['pass']

#Number of passing students
#print("Passing %d out of %d (%.2f%%)" % (np.sum(dataset_pass), len(dataset_pass), 100*float(np.sum(dataset_pass)) / len(dataset_pass)))

#fit decision tree
from sklearn import tree
for max_depth in range(1,20):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    t = t.fit(dataset_train_att, dataset_train_pass)
    #Score
    t.score(dataset_test_att, dataset_test_pass)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(t, dataset_att, dataset_pass, cv=5)
    print("Accuracy: %0.2f (+/_ %0.2f)" % (scores.mean(), scores.std() * 2))
#max_depth of 2 or 3 gives highest accuracy

depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1,20):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(t, dataset_att, dataset_pass, cv=5)
    depth_acc[i, 0] = max_depth
    depth_acc[i,1] = scores.mean()
    depth_acc[i, 2] = scores.std() * 2
    i += 1

#Make a nice presentation of this
fig, ax = plt.subplots()
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr = depth_acc[:,2])
plt.show()
