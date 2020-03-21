# -*- coding: utf-8 -*-
"""
Random forest model
"""
#Libraries
import pandas as pd
import numpy as np

#Load dataset
imgatt = pd.read_csv("CUB_200_2011/attributes/image_attribute_labels.txt", sep='\s+', header=None, error_bad_lines=False, 
                     warn_bad_lines=False, usecols=[0,1,2], names=['imgid', 'attid', 'present'])

#Currently we have 3.7mil rows. Organize so that each row is for an imgid with all the attributes (312).
imgatt2 = imgatt.pivot(index='imgid', columns='attid', values='present') 

#Load the image true classes.
imglabels = pd.read_csv("CUB_200_2011/image_class_labels.txt", sep=' ', header=None, names=['imgid', 'label'])
imglabels = imglabels.set_index('imgid')

#attach labels to attribute data set
df = imgatt2.join(imglabels)
df = df.sample(frac=1)
df_att = df.iloc[:, :312]
df_label = df.iloc[:,312:]

df_train_att = df_att[:8000]
df_test_att = df_att[8000:]
df_train_label = df_label[:8000]
df_test_label = df_label[8000:]
df_train_label = df_train_label['label']
df_test_label = df_test_label['label']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features=15, random_state=0, n_estimators=170)
#Each tree looks at 50 columns (max_features)
clf = clf.fit(df_train_att, df_train_label)

#Confusion matrix
from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(df_test_att)
cm = confusion_matrix(df_test_label, pred_labels)

#This code is from scikit-learn documentation.
import matplotlib.pyplot as plt
"""
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('COnfusion matrix, without normalization')
    
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    #fmt = '.2f' if normalize else 'd'
    #thresh = cm.max() / 2
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
birds = pd.read_csv("CUB_200_2011/classes.txt", sep='\s+', header=None, usecols=[1], names=['birdname'])
birds = birds['birdname']

np.set_printoptions(precision=2)
plt.figure(figsize=(60,60), dpi=300)
plot_confusion_matrix(cm, classes=birds, normalize=True)
plt.show()
"""
#This is a pretty good accuracy. It is for 100 trees with max_features of 50. 
#Next I will make a for loop that will change this a little bit and then we can see what gives the best accuracy

from sklearn.model_selection import cross_val_score 
"""
scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""

max_features_opts = range(5, 50, 5)
n_estimators_opts= range(10, 200, 20)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts), 4), float)
i = 0 
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, df_train_att, df_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i+= 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" % 
              (max_features, n_estimators, scores.mean(), scores.std() * 2))
        
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
fig.clf()
ax = fig.gca(projection='3d')
x = rf_params[:,0]
y = rf_params[:,1]
z = rf_params[:,2]
ax.scatter(x, y, z)
ax.set_zlim(0.2, 0.5)
ax.set_xlabel('Max Features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')
plt.show()
