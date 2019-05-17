# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:17:37 2019

@author: shaojie
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np
 
data_dir = 'MNIST_data/'
mnist = input_data.read_data_sets(data_dir,one_hot=False)
batch_size = 50000
batch_x,batch_y = mnist.train.next_batch(batch_size)
test_x = mnist.test.images[:10000]
test_y = mnist.test.labels[:10000]

im = np.array(test_x[0])
im = im.reshape(28, 28)
plt.imshow(im, cmap='gray')

print("start random forest")

for i in range(10,20,10):
    clf_rf = RandomForestClassifier(n_estimators=i)
    clf_rf.fit(batch_x,batch_y)
 
    y_pred_rf = clf_rf.predict(test_x)
    acc_rf = accuracy_score(test_y,y_pred_rf)
    print("n_estimators = %d, random forest accuracy:%f" %(i,acc_rf))

pred_0 = clf_rf.predict(test_x[0:2])
print(pred_0)