# -*- coding: utf-8 -*-
"""
Created on Mon oct 1 10:44:51 2018

@author: 
score card评分卡制作
"""

import copy
import numpy as np
import pandas as pd
import step1_woe
import step2_sklearn_model

df=step1_woe.df
dict_woe=step1_woe.scores
dict_woe1=copy.deepcopy(dict_woe)
#step1_woe.py脚本有问题，漏掉了account balance的woe，之后需要重新计算
#variable_names=list(df.columns),去掉target
variable_names=list(df.columns)[:-1]
coef=pd.read_excel("coef.xlsx")

B=step2_sklearn_model.B

df_total=pd.DataFrame()

for i in range(len(variable_names)):
    #依次遍历各个变量
    varName=variable_names[i]
    list_subVar_woe=dict_woe1[varName]
    #分箱个数
    numbers=len(list_subVar_woe)
    #依次变量每个变量的分箱，根据woe来计算score
    for number in range(numbers):
        var_Subdivision=list_subVar_woe[number]
        woe=var_Subdivision[1]
        coef1=float(coef[coef.variable_names==varName]['coef'])
        #变量评分公式
        score=woe*coef1*B*(-1)
        var_Subdivision.append(score)
        var_Subdivision.append(varName)

    names=['span','woe','score','varName']
    df_subVar_woe=pd.DataFrame(data=list_subVar_woe,columns=names)
    df_total=pd.concat([df_total,df_subVar_woe],axis=0)
        
df_total.to_excel("score_card.xlsx")

'''
#单个测试
varName=variable_names[0]
print("varName:",varName)
list_subVar_woe=dict_woe1[varName]
print("list_subVar_woe:",list_subVar_woe)
#分箱个数
numbers=len(list_subVar_woe)
print("number of bins:",numbers)
for number in range(numbers):
    var_Subdivision=list_subVar_woe[number]
    print("var_Subdivision:",var_Subdivision)
    woe=var_Subdivision[1]
    print("woe:",woe)
    coef1=float(coef[coef.variable_names==varName]['coef'])
    print("coef1:",coef1)
    #变量评分公式
    score=woe*coef1*B*(-1)
    print("score:",score)
    var_Subdivision.append(score)
    var_Subdivision.append(varName)
print ('last var_Subdivision:',var_Subdivision)
'''




