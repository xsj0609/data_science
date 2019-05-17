##############################################
### 对于连续性变量，直接分箱
### 定义一些分箱用到的辅助函数
##############################################

#当连续变量的初始取值集合太多时（>100），我们先对其进行初步划分
def SplitData(df, col, numOfSplit, special_attribute=[]):
    '''
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外
    :return: 在原数据集上增加一列，把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理
    '''
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = N//numOfSplit #每组样本数
    splitPointIndex = [i*n for i in range(1,numOfSplit)] #分割点的下标
    rawValues = sorted(list(df2[col])) #对取值进行排序
    splitPoint = [rawValues[i] for i in splitPointIndex] #分割点的取值
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint

#根据分割点将x映射至相应的组中
def AssignGroup(x, bin):
    '''
    :param x: 某个变量的某个取值
    :param bin: 上述变量的分箱结果
    :return: x在分箱结果下的映射
    '''
    N = len(bin)
    if x<=min(bin):
        return min(bin)
    elif x>max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]

def AssignBin(x, cutOffPoints,special_attribute=[]):
    '''
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x<=cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)

## 判断某变量的坏样本率是否单调，如果不单调，需缩减分箱个数
def BadRateMonotone(df, sortByVar, target,special_attribute = []):
    '''
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateNotMonotone = [badRate[i]<badRate[i+1] and badRate[i] < badRate[i-1] or badRate[i]>badRate[i+1] and badRate[i] > badRate[i-1]
                       for i in range(1,len(badRate)-1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


##############################################
### 计算卡方统计量
##############################################

def Chi2(df, total_col, bad_col, overallRate):
    '''
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param total_col: 全部样本的个数
    :param bad_col: 坏样本的个数
    :param overallRate: 全体样本的坏样本占比
    :return: 卡方值
    '''
    df2 = df.copy()
    # 期望坏样本个数＝全部样本个数*平均坏样本占比
    df2['good']=df2['total']-df2['bad']
    df2['expected_bad'] = df[total_col].apply(lambda x: x*overallRate)
    df2['expected_good']=df[total_col].apply(lambda x:x*(1-overallRate))
    combined_bad = zip(df2['expected_bad'], df2[bad_col])
    combined_good=zip(df2['expected_good'],df2['good'])
    chi_bad = [(i[0]-i[1])**2/i[0] for i in combined_bad]
    chi_good=[(i[0]-i[1])**2/i[0] for i in combined_good]
    df=(df2.shape[0]-1)*(2-1) #自由度
    chi2 = (sum(chi_bad)+sum(chi_good))/df
    return chi2

##############################################
### 卡方分箱
##############################################

### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals
def ChiMerge(df, col, target, max_interval=5,special_attribute=[],minBinPcnt=0):
    '''
    :param df: 包含目标变量与分箱属性的数据框
    :param col: 需要分箱的属性
    :param target: 目标变量，取值0或1
    :param max_interval: 最大分箱数。如果原始属性的取值个数低于该参数，不执行这段函数
    :param special_attribute: 不参与分箱的属性取值
    :param minBinPcnt：最小箱的占比，默认为0
    :return: 分箱结果
    '''
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)#不同的取值个数
    if N_distinct <= max_interval:  #如果原始属性的取值个数低于max_interval，不执行这段函数
        print ("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        if len(special_attribute)>=1:
            df1 = df.loc[df[col].isin(special_attribute)]
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))#该特征不同的取值

        # 步骤一: 通过col对数据集进行分组，求出每组的总样本数与坏样本数
        if N_distinct > 100:
            split_x = SplitData(df2, col, 100)
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        else:
            df2['temp'] = df2[col]
        # 总体bad rate将被用来计算expected bad count
        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)

        # 首先，每个单独的属性值将被分为单独的一组
        # 对属性值进行排序，然后两两组别进行合并
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]

        # 步骤二：建立循环，不断合并最优的相邻两个组别，直到：
        # 1，最终分裂出来的分箱数<＝预设的最大分箱数
        # 2，每箱的占比不低于预设值（可选）
        # 3，每箱同时包含好坏样本
        # 如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
        split_intervals = max_interval - len(special_attribute)
        while (len(groupIntervals) > split_intervals):  # 终止条件: 当前分箱数＝预设的分箱数
            # 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案
            chisqList = []
            for k in range(len(groupIntervals)-1):
                temp_group = groupIntervals[k] + groupIntervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = Chi2(df2b, 'total', 'bad', overallRate)
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined+1]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[best_comnbined+1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        # 检查是否有箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
        groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
        df2['temp_Bin'] = groupedvalues
        (binBadRate,regroup) = BinBadRate(df2, 'temp_Bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()),max(binBadRate.values())]
        while minBadRate ==0 or maxBadRate == 1:
            # 找出全部为好／坏样本的箱
            indexForBad01 = regroup[regroup['bad_rate'].isin([0,1])].temp_Bin.tolist()
            bin=indexForBad01[0]
            # 如果是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
            if bin == max(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[:-1]
            # 如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
            elif bin == min(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[1:]
            # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
            else:
                # 和前一箱进行合并，并且计算卡方值
                currentIndex = list(regroup.temp_Bin).index(bin)
                prevIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
                # 和后一箱进行合并，并且计算卡方值
                laterIndex = list(regroup.temp_Bin)[currentIndex + 1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])
            # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
            [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
        # 需要检查分箱后的最小占比
        if minBinPcnt > 0:
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            valueCounts = groupedvalues.value_counts().to_frame()
            valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / N)
            valueCounts = valueCounts.sort_index()
            minPcnt = min(valueCounts['pcnt'])
            while minPcnt < minBinPcnt and len(cutOffPoints) > 2:
                # 找出占比最小的箱
                indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0]
                # 如果占比最小的箱是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
                if indexForMinPcnt == max(valueCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                # 如果占比最小的箱是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
                elif indexForMinPcnt == min(valueCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                # 如果占比最小的箱是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
                else:
                    # 和前一箱进行合并，并且计算卡方值
                    currentIndex = list(valueCounts.index).index(indexForMinPcnt)
                    prevIndex = list(valueCounts.index)[currentIndex - 1]
                    df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                    chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
                    # 和后一箱进行合并，并且计算卡方值
                    laterIndex = list(valueCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                    chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])
        cutOffPoints = special_attribute + cutOffPoints
        return cutOffPoints

##############################################
### 
##############################################

# （iii）对连续型变量进行分箱，包括（ii）中的变量
continous_merged_dict = {}
for col in num_features:
    print ("{} is in processing".format(col))
    if -1 not in set(trainData[col]):   #－1会当成特殊值处理。如果没有－1，则所有取值都参与分箱
        max_interval = 5   #分箱后的最多的箱数
        cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval,special_attribute=[],minBinPcnt=0)
        trainData[col+'_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff,special_attribute=[]))
        monotone = BadRateMonotone(trainData, col+'_Bin', 'y')   # 检验分箱后的单调性是否满足
        while(not monotone):
            # 检验分箱后的单调性是否满足。如果不满足，则缩减分箱的个数。
            max_interval -= 1
            cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval, special_attribute=[],
                                          minBinPcnt=0)
            trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
            if max_interval == 2:
                # 当分箱数为2时，必然单调
                break
            monotone = BadRateMonotone(trainData, col + '_Bin', 'y')
        newVar = col + '_Bin'
        trainData[newVar] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
        var_bin_list.append(newVar)
    else:
        max_interval = 5
        # 如果有－1，则除去－1后，其他取值参与分箱
        cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval, special_attribute=[-1],
                                      minBinPcnt=0)
        trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
        monotone = BadRateMonotone(trainData, col + '_Bin', 'y',['Bin -1'])
        while (not monotone):
            max_interval -= 1
            # 如果有－1，－1的bad rate不参与单调性检验
            cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval, special_attribute=[-1],
                                          minBinPcnt=0)
            trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
            if max_interval == 3:
                # 当分箱数为3-1=2时，必然单调
                break
            monotone = BadRateMonotone(trainData, col + '_Bin', 'y',['Bin -1'])
        newVar = col + '_Bin'
        trainData[newVar] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
        var_bin_list.append(newVar)
    continous_merged_dict[col] = cutOff

##############################################
### WOE编码
##############################################

'''
第四步：WOE编码、计算IV
'''
WOE_dict = {}
IV_dict = {}
# 分箱后的变量进行编码，包括：
# 1，初始取值个数小于5，且不需要合并的类别型变量。存放在less_value_features中
# 2，初始取值个数小于5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
# 3，初始取值个数超过5，需要合并的类别型变量。合并后新的变量存放在var_bin_list中
# 4，连续变量。分箱后新的变量存放在var_bin_list中
all_var = var_bin_list  + less_value_features
for var in all_var:
    woe_iv = CalcWOE(trainData, var, 'y')
    WOE_dict[var] = woe_iv['WOE']
    IV_dict[var] = woe_iv['IV']

#将变量IV值进行降序排列，方便后续挑选变量
IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)

IV_values = [i[1] for i in IV_dict_sorted]
IV_name = [i[0] for i in IV_dict_sorted]
plt.title('feature IV')
plt.bar(range(len(IV_values)),IV_values)

##############################################
### 2.5 单变量分析和多变量分析
##############################################

'''
第五步：单变量分析和多变量分析，均基于WOE编码后的值。
（1）选择IV高于0.01的变量
（2）比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
'''

#选取IV>0.01的变量
high_IV = {k:v for k, v in IV_dict.items() if v >= 0.01}
high_IV_sorted = sorted(high_IV.items(),key=lambda x:x[1],reverse=True)

short_list = high_IV.keys()
short_list_2 = []
for var in short_list:
    newVar = var + '_WOE'
    trainData[newVar] = trainData[var].map(WOE_dict[var])
    short_list_2.append(newVar)

#对于上一步的结果，计算相关系数矩阵，并画出热力图进行数据可视化
trainDataWOE = trainData[short_list_2]
f, ax = plt.subplots(figsize=(10, 8))
corr = trainDataWOE.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)


#两两间的线性相关性检验
#1，将候选变量按照IV进行降序排列
#2，计算第i和第i+1的变量的线性相关系数
#3，对于系数超过阈值的两个变量，剔除IV较低的一个
deleted_index = []
cnt_vars = len(high_IV_sorted)
for i in range(cnt_vars):
    if i in deleted_index:
        continue
    x1 = high_IV_sorted[i][0]+"_WOE"
    for j in range(cnt_vars):
        if i == j or j in deleted_index:
            continue
        y1 = high_IV_sorted[j][0]+"_WOE"
        roh = np.corrcoef(trainData[x1],trainData[y1])[0,1]
        if abs(roh)>0.7:
            x1_IV = high_IV_sorted[i][1]
            y1_IV = high_IV_sorted[j][1]
            if x1_IV > y1_IV:
                deleted_index.append(j)
            else:
                deleted_index.append(i)

multi_analysis_vars_1 = [high_IV_sorted[i][0]+"_WOE" for i in range(cnt_vars) if i not in deleted_index]

'''
多变量分析：VIF
'''
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = np.matrix(trainData[multi_analysis_vars_1])
VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
max_VIF = max(VIF_list)
print (max_VIF)
# 最大的VIF是1.32267733123，因此这一步认为没有多重共线性
multi_analysis = multi_analysis_vars_1