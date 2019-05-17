def Chi2(df, total_col, bad_col,overallRate):
    '''
     #此函数计算卡方值
     :df dataFrame
     :total_col 每个值得总数量
     :bad_col 每个值的坏数据数量
     :overallRate 坏数据的占比
     : return 卡方值
    '''
    df2=df.copy()
    df2['expected']=df[total_col].apply(lambda x: x*overallRate)
    combined=zip(df2['expected'], df2[bad_col])
    chi=[(i[0]-i[1])**2/i[0] for i in combined]
    chi2=sum(chi)
    return chi2

#基于卡方阈值卡方分箱，有个缺点，不好控制分箱个数。
def ChiMerge_MinChisq(df, col, target, confidenceVal=3.841):
    '''
    #此函数是以卡方阈值作为终止条件进行分箱
    : df dataFrame
    : col 被分箱的特征
    : target 目标值,是0,1格式
    : confidenceVal  阈值，自由度为1， 自信度为0.95时，卡方阈值为3.841
    : return 分箱。
    这里有个问题，卡方分箱对分箱的数量没有限制，这样子会导致最后分箱的结果是分箱太细。
    '''
    #对待分箱特征值进行去重
    colLevels=set(df[col])
    
    #count是求得数据条数
    total=df.groupby([col])[target].count()
   
    total=pd.DataFrame({'total':total})
 
    #sum是求得特征值的和
    #注意这里的target必须是0,1。要不然这样求bad的数据条数，就没有意义，并且bad是1，good是0。
    bad=df.groupby([col])[target].sum()
    bad=pd.DataFrame({'bad':bad})
    #对数据进行合并，求出col，每个值的出现次数（total，bad）
    regroup=total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
  
    #求出整的数据条数
    N=sum(regroup['total'])
    #求出黑名单的数据条数
    B=sum(regroup['bad'])
    overallRate=B*1.0/N
    
    #对待分箱的特征值进行排序
    colLevels=sorted(list(colLevels))
    groupIntervals=[[i] for i in colLevels]
   
    groupNum=len(groupIntervals)
    while(1):
        if len(groupIntervals) == 1:
            break
        chisqList=[]
        for interval in groupIntervals:
            df2=regroup.loc[regroup[col].isin(interval)]
            chisq=Chi2(df2, 'total', 'bad', overallRate)
            chisqList.append(chisq)

        min_position=chisqList.index(min(chisqList))
    
        if min(chisqList) >= confidenceVal:
            break
        
        if min_position==0:
            combinedPosition=1
        elif min_position== groupNum-1:
            combinedPosition=min_position-1
        else:
            if chisqList[min_position-1]<=chisqList[min_position + 1]:
                combinedPosition=min_position-1
            else:
                combinedPosition=min_position+1
        groupIntervals[min_position]=groupIntervals[min_position]+groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum=len(groupIntervals)
    return groupIntervals

#最大分箱数分箱
def ChiMerge_MaxInterval_Original(df, col, target,max_interval=5):
    '''
    : df dataframe
    : col 要被分项的特征
    ： target 目标值 0,1 值
    : max_interval 最大箱数
    ：return 箱体
    '''
    colLevels=set(df[col])
    colLevels=sorted(list(colLevels))
    N_distinct=len(colLevels)
    if N_distinct <= max_interval:
        print("the row is cann't be less than interval numbers")
        return colLevels[:-1]
    else:
        total=df.groupby([col])[target].count()
        total=pd.DataFrame({'total':total})
        bad=df.groupby([col])[target].sum()
        bad=pd.DataFrame({'bad':bad})
        regroup=total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N=sum(regroup['total'])
        B=sum(regroup['bad'])
        overallRate=B*1.0/N
        groupIntervals=[[i] for i in colLevels]
        groupNum=len(groupIntervals)
        while(len(groupIntervals)>max_interval):
            chisqList=[]
            for interval in groupIntervals:
                df2=regroup.loc[regroup[col].isin(interval)]
                chisq=Chi2(df2,'total','bad',overallRate)
                chisqList.append(chisq)
            min_position=chisqList.index(min(chisqList))
            if min_position==0:
                combinedPosition=1
            elif min_position==groupNum-1:
                combinedPosition=min_position-1
            else:
                if chisqList[min_position-1]<=chisqList[min_position + 1]:
                    combinedPosition=min_position-1
                else:
                    combinedPosition=min_position+1
            #合并箱体
            groupIntervals[min_position]=groupIntervals[min_position]+groupIntervals[combinedPosition]
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum=len(groupIntervals)
        groupIntervals=[sorted(i) for i in groupIntervals]
        print(groupIntervals)
        cutOffPoints=[i[-1] for i in groupIntervals[:-1]]
        return cutOffPoints

#计算WOE和IV值
def CalcWOE(df,col, target):
    '''
    : df dataframe
    : col 注意这列已经分过箱了，现在计算每箱的WOE和总的IV
    ：target 目标列 0-1值
    ：return 返回每箱的WOE和总的IV
    '''
    total=df.groupby([col])[target].count()
    total=pd.DataFrame({'total':total})
    bad=df.groupby([col])[target].sum()
    bad=pd.DataFrame({'bad':bad})
    regroup=total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N=sum(regroup['total'])
    B=sum(regroup['bad'])
    regroup['good']=regroup['total']-regroup['bad']
    G=N-B
    regroup['bad_pcnt']=regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt']=regroup['good'].map(lambda x: x*1.0/G)
    regroup['WOE']=regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis=1)
    WOE_dict=regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    IV=regroup.apply(lambda x:(x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis=1)
    IV_SUM=sum(IV)
    return {'WOE':WOE_dict,'IV_sum':IV_SUM,'IV':IV}

#分箱以后检查每箱的bad_rate的单调性，如果不满足，那么继续进行相邻的两项合并，直到bad_rate单调为止
def BadRateMonotone(df, sortByVar, target):
    #df[sortByVar]这列已经经过分箱
    df2=df.sort_values(by=[sortByVar])
    total=df2.groupby([sortByVar])[target].count()
    total=pd.DataFrame({'total':total})
    bad=df2.groupby([sortByVar])[target].sum()
    bad=pd.DataFrame({'bad':bad})
    regroup=total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined=zip(regroup['total'], regroup['bad'])
    badRate=[x[1]*1.0/x[0] for x in combined]
    badRateMonotone=[badRate[i]<badRate[i+1] for i in range(len(badRate)-1)]
    Monotone = len(set(badRateMonotone))
    if Monotone==1:
        return True
    else:
        return False

 #检查最大箱，如果最大箱里面数据数量占总数据的90%以上，那么弃用这个变量
def MaximumBinPcnt(df, col):
    N=df.shape[0]
    total=df.groupby([col])[col].count()
    pcnt=total*1.0/N
    return max(pcnt)

#对于类别型数据，以bad_rate代替原有值，转化成连续变量再进行分箱计算。比如我们这里的户籍地代码，就是这种数据格式
#当然如果类别较少时，原则上不需要分箱
def BadRateEncoding(df, col, target):
    '''
    : df DataFrame
    : col 需要编码成bad rate的特征列
    ：target值，0-1值
    ： return: the assigned bad rate 
    '''
    total=df.groupby([col])[target].count()
    total=pd.DataFrame({'total':total})
    bad=df.groupby([col])[target].sum()
    bad=pd.DataFrame({'bad':bad})
    regroup=total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate']=regroup.apply(lambda x: x.bad*1.0/x.total, axis=1)
    br_dict=regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    badRateEnconding=df[col].map(lambda x: br_dict[x]['bad_rate'])
    return {'encoding':badRateEnconding,'br_rate':br_dict}

class Woe_IV:


    def __init__(self,df,colList,target):
        '''
        :param df: 这个是用来分箱的dataframe
        :param colList: 这个分箱的列数据，数据结构是一个字段数组
         例如colList=[
              {
                'col':'openning_room_num_n3'
                'bandCol':'openning_room_num_n3_band',
                'bandNum':6,
                ‘toCsvPath':'/home/liuweitang/yellow_model/data/mk/my.txt'
              },

         ]
        :param target 目标列0-1值，1表示bad，0表示good
        '''
        self.df=df
        self.colList=colList
        self.target=target

    def to_band(self):
        for i in range(len(self.colList)):
            colParam=self.colList[i]
            #计算出箱体分别值，返回的是一个长度为5数组[0,4,13,45,78]或者长度为6的数组[0,2,4,56,67,89]
            cutOffPoints=ChiMerge_MaxInterval_Original(self.df,colParam['col'],self.target,colParam['bandNum'])
            print(cutOffPoints)
            
            indexValue=0
            value_band=[]
            #那么cutOffPoints第一个值就是作为一个独立的箱
            if len(cutOffPoints) == colParam['bandNum']-1:
                print('len-1 type')
                for i in range(0,len(cutOffPoints)):
                    if i==0:
                        self.df.loc[self.df[colParam['col']]<=cutOffPoints[i], colParam['bandCol']]=indexValue
                        indexValue+=1
                        value_band.append('0-'+str(cutOffPoints[i]))
                    if 0<i<len(cutOffPoints):
                        self.df.loc[(self.df[colParam['col']] > cutOffPoints[i - 1]) & (self.df[colParam['col']] <= cutOffPoints[i]), colParam['bandCol']] = indexValue
                        indexValue+=1
                        value_band.append(str(cutOffPoints[i - 1]+1)+"-"+str(cutOffPoints[i]))
                    if i==len(cutOffPoints)-1:
                        self.df.loc[self.df[colParam['col']] > cutOffPoints[i], colParam['bandCol']] = indexValue
                        value_band.append(str(cutOffPoints[i]+1)+"-")

            #那么就是直接分割分箱，
            if len(cutOffPoints)==colParam['bandNum']:
                print('len type')
                for i in range(0,len(cutOffPoints)):
                    if 0< i < len(cutOffPoints):
                        self.df.loc[(self.df[colParam['col']] > cutOffPoints[i - 1]) & (self.df[colParam['col']] <= cutOffPoints[i]), colParam['bandCol']] = indexValue
                        value_band.append(str(cutOffPoints[i - 1]+1)+"-"+str(cutOffPoints[i]))
                        indexValue += 1
                    if i == len(cutOffPoints)-1:
                        self.df.loc[self.df[colParam['col']] > cutOffPoints[i], colParam['bandCol']] = indexValue
                        value_band.append(str(cutOffPoints[i]+1)+"-")
                        
            self.df[colParam['bandCol']].astype(int)
            #到此分箱结束，下面判断单调性
            isMonotone = BadRateMonotone(self.df,colParam['bandCol'], self.target)

            #如果不单调，那就打印出错误，并且继续执行下一个特征分箱
            if isMonotone==False:
                print(colParam['col']+' band error, reason is not monotone')
                continue

            #单调性判断完之后，就要计算woe_IV值
            woe_IV=CalcWOE(self.df, colParam['bandCol'],self.target)
            woe=woe_IV['WOE']
            woe_result=[]
            for i in range(len(woe)):
                woe_result.append(woe[i]['WOE'])
            
            iv=woe_IV['IV']
            iv_result=[]
            for i in range(len(iv)):
                iv_result.append(iv[i])
                
            good_bad_count=self.df.groupby([colParam['bandCol'],self.target]).label.count()
            good_count=[]
            bad_count=[]
            for i in range(0,colParam['bandNum']):
                good_count.append(good_bad_count[i][0])
                bad_count.append(good_bad_count[i][1])
            
            print(value_band)
            print(good_count)
            print(bad_count)
            print(woe_result)
            print(iv_result)
            #将WOE_IV值保存为dataframe格式数据，然后导出到csv
            #这里其实还有个问题，就是
            woe_iv_df=pd.DataFrame({
                'IV':iv_result,
                'WOE':woe_result,
                'bad':bad_count,
                'good':good_count,
                colParam['bandCol']:value_band
            })
            bad_good_count=self.df.groupby([colParam['bandCol'],self.target])[self.target].count();
           
            woe_iv_df.to_csv(colParam['toCsvPath'])
            print(colParam['col']+'band finished')

openning_data=pd.read_csv('***',sep='$')
colList=[
    {
        'col':'openning_room_0_6_num_n3',
        'bandCol':'openning_room_0_6_num_n3_band',
        'bandNum':5,
        'toCsvPath':'/home/liuweitang/yellow_model/eda/band_result/openning_room_0_6_num_n3_band.csv'
    },
    {
        'col':'openning_room_6_12_num_n3',
        'bandCol':'openning_room_6_12_num_n3_band',
        'bandNum':5,
        'toCsvPath':'/home/liuweitang/yellow_model/eda/band_result/openning_room_6_12_num_n3_band.csv'
    }
]
band2=Woe_IV(openning_data,colList,'label')
band2.to_band()