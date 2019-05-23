# -*- coding: utf-8 -*-
"""
@author
"""
import pickle
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
#混淆矩阵计算
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

df_german=pd.read_excel("german_woe.xlsx")
#df_german=pd.read_excel("german_credit.xlsx")
#df_german=pd.read_excel("df_after_vif.xlsx")
y=df_german["target"]
#x=df_german.ix[:,"Account Balance":"Foreign Worker"]
x=df_german.ix[:,"Account Balance":"Foreign Worker"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

#验证
print("accuracy on the training subset:{:.3f}".format(classifier.score(X_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(classifier.score(X_test,y_test)))

#得分公式
'''
P0 = 50
PDO = 10
theta0 = 1.0/20
B = PDO/np.log(2)
A = P0 + B*np.log(theta0)
'''
def Score(probability):
    #底数是e
    score = A-B*np.log(probability/(1-probability))
    return score
#批量获取得分
def List_score(pos_probablity_list):
    list_score=[]
    for probability in pos_probablity_list:
        score=Score(probability)
        list_score.append(score)
    return list_score

P0 = 50
PDO = 10
theta0 = 1.0/20
B = PDO/np.log(2)
A = P0 + B*np.log(theta0)
print("A:",A)
print("B:",B)
list_coef = list(classifier.coef_[0])
intercept= classifier.intercept_

#获取所有x数据的预测概率,包括好客户和坏客户，0为好客户，1为坏客户
probablity_list=classifier.predict_proba(x)
#获取所有x数据的坏客户预测概率
pos_probablity_list=[i[1] for i in probablity_list]
#获取所有客户分数
list_score=List_score(pos_probablity_list)
list_predict=classifier.predict(x)
df_result=pd.DataFrame({"label":y,"predict":list_predict,"pos_probablity":pos_probablity_list,"score":list_score})

df_result.to_excel("score_proba.xlsx")

#变量名列表
list_vNames=df_german.columns
#去掉第一个变量名target
list_vNames=list_vNames[1:]
df_coef=pd.DataFrame({"variable_names":list_vNames,"coef":list_coef})
df_coef.to_excel("coef.xlsx")


#保存模型
save_classifier = open("scoreCard.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()