#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:50:53 2019

@author: xushaojie
"""
trainData = pd.read_csv('训练集.csv',header = 0,engine ='python')
testData = pd.read_csv('测试集.csv',header = 0,engine ='python')

##@ In[3]:


# 衍生逾期类型的特征的函数
def DelqFeatures(event,window,type):
    '''
    :parms event 数据框
    :parms windows 时间窗口
    :parms type 响应事件类型
    '''
    current = 12
    start = 12 - window + 1
    #delq1、delq2、delq3为了获取window相对应的dataframe范围
    delq1 = [event[a] for a in ['Delq1_' + str(t) for t in range(current, start - 1, -1)]]
    delq2 = [event[a] for a in ['Delq2_' + str(t) for t in range(current, start - 1, -1)]]
    delq3 = [event[a] for a in ['Delq3_' + str(t) for t in range(current, start - 1, -1)]]
    if type == 'max delq':
        if max(delq3) == 1:
            return 3
        elif max(delq2) == 1:
            return 2
        elif max(delq1) == 1:
            return 1
        else:
            return 0
    if type in ['M0 times','M1 times', 'M2 times']:
        if type.find('M0')>-1:
            return sum(delq1)
        elif type.find('M1')>-1:
            return sum(delq2)
        else:
            return sum(delq3)

allFeatures = []
 
'''
逾期类型的特征在行为评分卡（预测违约行为）中，一般是非常显著的变量。
通过设定时间窗口，可以衍生以下类型的逾期变量：
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    # 1，过去t时间窗口内的最大逾期状态
    allFeatures.append('maxDelqL'+str(t)+"M")
    trainData['maxDelqL'+str(t)+"M"] = trainData.apply(lambda x: DelqFeatures(x,t,'max delq'),axis=1)
 
    # 2，过去t时间窗口内的，M0,M1,M2的次数
    allFeatures.append('M0FreqL' + str(t) + "M")
    trainData['M0FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x,t,'M0 times'),axis=1)
 
    allFeatures.append('M1FreqL' + str(t) + "M")
    trainData['M1FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M1 times'), axis=1)
 
    allFeatures.append('M2FreqL' + str(t) + "M")
    trainData['M2FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M2 times'), axis=1)


##@ In[4]:


#衍生额度使用率类型特征的函数
def UrateFeatures(event, window, type):
    '''
    :parms event 数据框
    :parms windows 时间窗口
    :parms type 响应事件类型
    '''
    current = 12
    start = 12 - window + 1
    #获取在数据框内有效区域
    monthlySpend = [event[a] for a in ['Spend_' + str(t) for t in range(current, start - 1, -1)]]
    #获取授信总额度
    limit = event['Loan_Amount']
    #月使用率
    monthlyUrate = [x / limit for x in monthlySpend]
    if type == 'mean utilization rate':
        return np.mean(monthlyUrate)
    if type == 'max utilization rate':
        return max(monthlyUrate)
    #月额度使用率增加的月份
    if type == 'increase utilization rate':
        #val[0:-1]表示第一个元素到倒数第二个元素的切片
        currentUrate = monthlyUrate[0:-1]
        #val[1:]表示第二个元素到最后一个元素的切片
        previousUrate = monthlyUrate[1:]
        compareUrate = [int(x[0]>x[1]) for x in zip(currentUrate,previousUrate)]
        return sum(compareUrate)

'''
额度使用率类型特征在行为评分卡模型中，通常是与违约高度相关的
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    # 1，过去t时间窗口内的最大月额度使用率
    allFeatures.append('maxUrateL' + str(t) + "M")
    trainData['maxUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x,t,'max utilization rate'),axis = 1)
 
    # 2，过去t时间窗口内的平均月额度使用率
    allFeatures.append('avgUrateL' + str(t) + "M")
    trainData['avgUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x, t, 'mean utilization rate'),axis=1)
 
    # 3，过去t时间窗口内，月额度使用率增加的月份。该变量要求t>1
    if t > 1:
        allFeatures.append('increaseUrateL' + str(t) + "M")
        trainData['increaseUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x, t, 'increase utilization rate'),axis=1)


##@ In[5]:


#衍生还款类型特征的函数
def PaymentFeatures(event, window, type):
    current = 12
    start = 12 - window + 1
    #月还款金额
    currentPayment = [event[a] for a in ['Payment_' + str(t) for t in range(current, start - 1, -1)]]
    #月使用金额，错位一下
    previousOS = [event[a] for a in ['OS_' + str(t) for t in range(current-1, start - 2, -1)]]
    monthlyPayRatio = []
    for Pay_OS in zip(currentPayment,previousOS):
        #前一个月使用了才会产生还款
        if Pay_OS[1]>0:
            payRatio = Pay_OS[0]*1.0 / Pay_OS[1]
            monthlyPayRatio.append(payRatio)
        #前一个月没使用，就按照100%还款
        else:
            monthlyPayRatio.append(1)
    if type == 'min payment ratio':
        return min(monthlyPayRatio)
    if type == 'max payment ratio':
        return max(monthlyPayRatio)
    if type == 'mean payment ratio':
        total_payment = sum(currentPayment)
        total_OS = sum(previousOS)
        if total_OS > 0:
            return total_payment / total_OS
        else:
            return 1

'''
还款类型特征也是行为评分卡模型中常用的特征
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    # 1，过去t时间窗口内的最大月还款率
    allFeatures.append('maxPayL' + str(t) + "M")
    trainData['maxPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'max payment ratio'),axis=1)
    # 2，过去t时间窗口内的最小月还款率
    allFeatures.append('minPayL' + str(t) + "M")
    trainData['minPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'min payment ratio'),axis=1)
    # 3，过去t时间窗口内的平均月还款率
    allFeatures.append('avgPayL' + str(t) + "M")
    trainData['avgPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'mean payment ratio'),axis=1)
