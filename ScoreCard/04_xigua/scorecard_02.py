# -*- coding: utf-8 -*-

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pandas as pd
import numpy as np
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体    
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
try:
    import cPickle as pickle
except ImportError:
    import pickle
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os

os.chdir('D:/Server/NutsCloud/data_science/ScoreCard/04_xigua/')

pd.set_option('display.max_columns', 100000)  
pd.set_option('display.max_rows', 10000)

class XXDNumberBin():
    '''
    分箱类
    '''
    def __init__(self):
        self.__bin_stats = None
        
    def get_bin_stats(self):
        if self.__bin_stats is not None:
            return self.__bin_stats.reset_index(drop=True)
    
    def get_cutoff(self):
        if self.__bin_stats is not None:
            return self.__bin_stats.Max.dropna().tolist()
    
    def trans_bin_to_woe(self,B):
        '''
        
        B: Series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        woe =  self.__bin_stats['WoE'].sort_index()
        return B.map(lambda x:woe[x])
    
    def plot_woe(self,title=None):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        woe =  self.__bin_stats[['WoE','Range']].sort_index()
        plt.clf()
        if title is None:
            title = self.__varname
        plt.title('{}(WOE)'.format(title))
        plt.bar(range(len(woe)), woe.WoE,tick_label=woe.Range)
        plt.show()
        print('Cutoff:{}'.format(self.get_cutoff()))
        
    def get_iv(self):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        return self.__bin_stats['TotalIV'].iloc[0]
    
    def get_varname(self):
        return self.__varname;
    
    def trans_to_bin(self,X):
        '''
        如果训练集有缺失：
        1）缺失值分到缺失组，
        2）小于最小值的分到第一组
        3) 超过最大值的分最后一组。
        如果训练集没有缺失：
        1）缺失值\小于最小值分到第一组；
        2）超过最大值的分最后一组
        
        X: series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        if not is_numeric_dtype(X):
            X = X.astype(float)
        cuts = self.__bin_stats['Max'].sort_values(na_position ='first')
        mx = cuts.max()
        return X.map(lambda x:(cuts>=x).idxmax() if x<=mx else cuts.index[-1],na_action='ignore').fillna(cuts.index[0])
    
    def trans_to_woe(self,X):
        '''
        如果训练集有缺失：
        1）缺失值分到缺失组，
        2）小于最小值的分到第一组
        3) 超过最大值的分最后一组。
        如果训练集没有缺失：
        1）缺失值\小于最小值分到第一组；
        2）超过最大值的分最后一组
        X : series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        if not is_numeric_dtype(X):
            X = X.astype(float)
        cuts = self.__bin_stats['Max'].sort_values(na_position ='first')
        mx = cuts.max()
        woe =  self.__bin_stats['WoE'].sort_index()
        return X.map(lambda x:woe[(cuts>=x).idxmax()] if x<=mx else woe.iloc[-1] ,na_action='ignore').fillna(woe.iloc[0])
    
    def __cc(self,dfx):
            mx=dfx.XX.max()
            mn=dfx.XX.min()
            cnt=len(dfx)
            bad=dfx.YY.sum()
            good=cnt-bad            
            return pd.Series({'Var':self.__varname,'Range':'<={:.3f}'.format(mx) if pd.notna(mx) else 'Miss',
                              'Min':mn, 'Max':mx,'CntRec':cnt,'CntGood':good,'CntBad':bad})
        
    def calc_stats(self,data):
        '''
        计算woe，iv等。
        data: df[['bin','XX',YY']]
        '''
        res = data.groupby(data['bin']).apply(self.__cc)      
        cntg= (data.YY==0).sum()
        cntb= (data.YY==1).sum()
        res['Pct']=res.CntRec/len(data)
        res['PctBad']=res.CntBad/cntb
        res['PctGood']=res.CntGood/cntg
        res['BadRate']=res.CntBad/res.CntRec
        res['CumGood']=res.CntGood.cumsum()
        res['CumBad']=res.CntBad.cumsum()
        res['Odds']=res.BadRate/(1-res.BadRate)
        res['LnOdds']=np.log(res.Odds)
        woe = np.log(res.PctBad/res.PctGood)
        woe[res.PctBad==0] = 20
        woe[res.PctGood==0] = -20
        res['WoE'] = woe
        res['IV'] = (res.PctBad-res.PctGood)*res.WoE
        res['TotalIV'] = res.IV[(res.PctBad>0)&(res.PctGood>0)].sum()
        #res=res.append(pd.Series({'Var':x,'Min':XX.min(),'Max':XX.max(),'LnOdds':np.log(),'IV':res.IV.sum()},name='ALL'))
        return res
    def manual_bin(self,df,x,y,cutoff=[]):
        '''
        手动分箱
        df: 数据
        x: 变量名
        y: 目标变量
        '''
        self.__varname = x
        XX,YY = df[x],df[y]
        assert YY.isin([0,1]).all(),'ERROR: {} 目标变量非0/1!'.format(y)
        if not is_numeric_dtype(XX):
            XX = XX.astype(float) 
        data = pd.DataFrame({'XX':XX,'YY':YY})
        cnt = XX.count()
        assert cnt>0,'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        edges = pd.Series(cutoff+[np.inf]).sort_values()
        mx = edges.max()
        data['bin'] = XX.map(lambda x:(edges>=x).idxmax() if x<=mx else edges.index[-1],na_action='ignore').fillna(-1)
        self.__bin_stats= self.calc_stats(data)
        
    def pct_bin(self,df,x,y,max_bin=10,min_pct=0.06):        
        '''
        等频分箱。
        df: 数据
        x: 变量名
        y: 目标变量
        '''
        self.__varname = x
        XX,YY = df[x],df[y]
        assert YY.isin([0,1]).all(),'ERROR: {}  目标变量非0/1!'.format(y)
        if not is_numeric_dtype(XX):
            XX = XX.astype(float) 
        data = pd.DataFrame({'XX':XX,'YY':YY})
        cnt = XX.count()
        assert cnt>0,'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        min_sample = int(len(XX)*min_pct)
        if cnt<= min_sample:
            print('WARN: "{}" 非空值少于 {} !'.format(x,min_pct))
        nuniq = XX.nunique()
        if nuniq<= 50:
            print('WARN: "{}" 数值型变量只有 {} 个取值!'.format(x,nuniq))
        cut_ok = False
        ZZ = XX.rank(pct=1)
        while not cut_ok:
            edges = pd.Series(np.linspace(0,1,max_bin+1))
            bins =ZZ.map(lambda r:(edges>=r).idxmax(),na_action='ignore').fillna(-1)
            cut_ok = True
            if bins.value_counts().min() < min_sample and cnt>min_sample and max_bin>1:
                max_bin=max_bin-1
                cut_ok=False
        data['bin']=bins
        self.__bin_stats= self.calc_stats(data)
        
    def monotone_bin(self,df,x,y,max_bin=10):
        '''
        单调分箱。
        df: 数据
        x: 变量名
        y: 目标变量
        '''
        self.__varname = x
        XX,YY = df[x],df[y]
        assert YY.isin([0,1]).all(),'ERROR: {} 目标变量非0/1!'.format(y)
        if not is_numeric_dtype(XX):
            XX = XX.astype(float) 
        data = pd.DataFrame({'XX':XX,'YY':YY})
        cnt = XX.count()
        assert cnt>0,'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        cut_ok = False
        ZZ = XX.rank(pct=1)
        while not cut_ok:
            edges = pd.Series(np.linspace(0,1,max_bin+1))
            data['bin']=ZZ.map(lambda r:(edges>=r).idxmax(),na_action='ignore').fillna(-1)
            res=self.calc_stats(data).sort_index()
            woe = res[~res.Max.isna()].WoE
            cut_ok = woe.is_monotonic_decreasing or woe.is_monotonic_increasing
            max_bin = max_bin-1
        self.__bin_stats= res
    

class XXDCharBin():
    def __init__(self):
        self.__bin_stats=None
    
    def get_bin_stats(self):
        if self.__bin_stats is not None:
            return self.__bin_stats.copy()
        
    def trans_bin_to_woe(self,B):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        data = B.to_frame()
        woe =  self.__bin_stats['WoE'].sort_index()
        return B.map(lambda x:woe[x],na_action='ignore').fillna(woe.iloc[0])
    
    def plot_woe(self,title=None):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        woe =  self.__bin_stats[['WoE','Range']].sort_values(by='WoE')
        plt.clf()
        if title is None:
            title = self.__varname
        plt.title('{}(WOE)'.format(title))
        plt.bar(range(len(woe)), woe.WoE)
        plt.show()
        print(woe.Range.reset_index(drop=True))
        
    def get_iv(self):
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        return self.__bin_stats['TotalIV'].iloc[0]
    
    def get_varname(self):
        return self.__varname;
    
    def trans_to_bin(self,X):
        '''
        新值分到缺失
        X: series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        if not is_string_dtype(X):
            X = X.astype(str)
        data = X.to_frame()
        data['bin'] = -1
        for bin,values in enumerate(self.__bins):
            data.loc[X.isin(values),'bin']=bin
        return data['bin']
    
    def trans_to_woe(self,X):
        '''
        新值分到缺失
        X: series
        '''
        if self.__bin_stats is None:
            raise ValueError('ERROR: 尚未调用分箱函数，无法转换!')
        if not is_string_dtype(X):
            X = X.astype(str)
        data = X.to_frame()
        woe =  self.__bin_stats['WoE'].sort_index()
        data['woe'] = woe.iloc[0]
        for bin,values in enumerate(self.__bins):
            data.loc[X.isin(values),'woe']=woe[bin]
        return data['woe']
    
    def __cc(self,dfx):
            cnt=len(dfx)
            bad=dfx.YY.sum()
            good=cnt-bad            
            return pd.Series({'Var':self.__varname,'Range':dfx.XX.unique(),'CntRec':cnt,'CntGood':good,'CntBad':bad})
        
    def calc_stats(self,data):
        '''
        计算woe，iv等。
        '''
        res = data.groupby(data['bin']).apply(self.__cc)      
        cntg= (data.YY==0).sum()
        cntb= (data.YY==1).sum()
        res['Pct']=res.CntRec/len(data)
        res['PctBad']=res.CntBad/cntb
        res['PctGood']=res.CntGood/cntg
        res['BadRate']=res.CntBad/res.CntRec
        res['CumGood']=res.CntGood.cumsum()
        res['CumBad']=res.CntBad.cumsum()
        res['Odds']=res.BadRate/(1-res.BadRate)
        res['LnOdds']=np.log(res.Odds)
        woe = np.log(res.PctBad/res.PctGood)
        woe[res.PctBad==0] = 20
        woe[res.PctGood==0] = -20
        res['WoE'] = woe
        res['IV'] = (res.PctBad-res.PctGood)*res.WoE
        res['TotalIV'] = res.IV[(res.PctBad>0)&(res.PctGood>0)].sum()
        return res
    def manual_bin(self,df,x,y,bins=[]):
        '''
        手动分箱
        df: 数据
        x: 变量名
        y: 目标变量
        bins: [['a'],['b'],['c','d'],['e']]
        '''
        self.__varname = x
        data = pd.DataFrame({'XX':df[x],'YY':df[y]})
        assert data.YY.isin([0,1]).all(),'ERROR: {} 目标变量非0/1!'.format(y)
        if not is_string_dtype(data.XX):
            data['XX'] = data.XX.astype(str) 
        cnt = data.XX.count()
        assert cnt>0,'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        data['bin'] = -1
        for i,values in enumerate(bins):
            data.loc[data.XX.isin(values),'bin']=i
        data.loc[data.XX.isnull(),'bin'] = -2
        self.__bins=bins.copy()
        res = self.calc_stats(data)
        self.__bin_stats= res
        
    def pct_bin(self,df,x,y,sp_bins = [],max_bin=10): 
        '''
        字符型自动分箱，
        sp_bins: 特殊值分箱. [['a'],['b'],['c','d'],['e']]
        df: 数据
        x: 变量名
        y: 目标变量
        '''
        spvars = []
        for binb in sp_bins:
                spvars = spvars + binb
        assert len(set(spvars))==len(spvars),'ERROR: "{}" : sp_bins are overlapping!'.format(x)
        data = pd.DataFrame({'XX':df[x],'YY':df[y]})
        assert data.YY.isin([0,1]).all(),'ERROR: {} 目标变量非0/1!'.format(y)
        data = data.dropna()
        cnt = data.shape[0]
        assert cnt>0,'ERROR: "{}" 变量值全为 NULL  !'.format(x)
        if not is_string_dtype(data.XX):
            data['XX'] = data.XX.astype(str) 
        nuniq = data.XX.nunique()
        if nuniq> 50:
            print('WARN: "{}" 字符型变量取值数超过 {} 个!'.format(x,nuniq))
    
        db = data[~data.XX.isin(spvars)]
        dbr=db.groupby('XX').YY.mean().reset_index()
        dbr['rr'] = dbr.YY.rank(pct=1)
        edges = pd.Series(np.linspace(0,1,max_bin+1))
        dbr['bin'] =dbr.rr.map(lambda r:(edges>=r).idxmax())
        xx = dbr.groupby('bin').apply(lambda yy:yy.XX.tolist())
        sp_bins = sp_bins +xx.tolist()
        self.manual_bin(df,x,y,sp_bins.copy())
        

class XXDBinUtils():
    '''
    批量分箱类
    '''
    def __init__(self,df,y):
        self.__df = df
        self.__y=y
        self.__bins={}
    def auto_bin(self,ignore=[],params={}):
        for col in self.__df.columns:
            if col == self.__y or col in ignore:
                continue
            max_bin = params['max_bin'] if 'max_bin' in params else 10
            min_pct = params['min_pct'] if 'min_pct' in params else 0.06
            print('now binning {}...'.format(col))
            if is_numeric_dtype(self.__df[col]):
                nb = XXDNumberBin()
                nb.pct_bin(self.__df,col,self.__y,max_bin=max_bin,min_pct=min_pct)
                self.__bins[col] = nb
                
            else: 
                cb = XXDCharBin()
                cb.pct_bin(self.__df,col,self.__y,max_bin=max_bin)
                self.__bins[col] = cb
        print('done.')
        
    def manual_bin(self,col,bins_or_cutoff):
        assert col != self.__y,'ERROR: 不能对目标变量分箱！'
        if col in self.__bins:
            self.__bins[col].manual_bin(self.__df,col,self.__y,bins_or_cutoff)
        elif is_numeric_dtype(self.__df[col]):
                nb = XXDNumberBin()
                nb.manual_bin(self.__df,col,self.__y,bins_or_cutoff)
                self.__bins[col] = nb
                
        else: 
                cb = XXDCharBin()
                cb.manual_bin(self.__df,col,self.__y,bins_or_cutoff)
                self.__bins[col] = cb
        print('done.')
    def get_bin(self,col):
        return self.__bins.get(col)
    
    def trans_to_woe(self,df,inplace=False):
        res = pd.DataFrame()
        if inplace:
            res = df
        for col in df.columns:
            if col in self.__bins:
                res['woe_{}'.format(col)] = self.__bins[col].trans_to_woe(df[col])
        return res
    
    def get_all_bin_stats(self):
        res = pd.DataFrame()
        for col in self.__bins:
            res = res.append(self.__bins[col].get_bin_stats())
        return res
    
    def dump(self,filename):
        f = open(filename, 'wb')
        pickle.dump(self.__bins,f)
        f.close()
        print('done.')
        
    def load(self,filename):
        f = open(filename, 'rb')
        d = pickle.load(f)
        f.close()
        self.__bins = d
        print('done.')
        
    def iv_ks_auc(self):
        result = []
        y = self.__df[self.__y].values
        for col in self.__bins:
                woe = self.__bins[col].trans_to_woe(self.__df[col])
                iv = self.__bins[col].get_iv()
                ks = stats.ks_2samp(woe[y==0], woe[y==1]).statistic
                auc = metrics.roc_auc_score(y,woe)
                result.append((col,iv,ks,auc))
        return pd.DataFrame(result,columns=['var','iv','ks','auc'])
    
    def plot_woe(self,col,title=None):
        if col in self.__bins:
            self.__bins[col].plot_woe(title)
    def plot_all_woe(self,min_iv=0.0,max_iv=10.0):
        '''

        :param min_iv:
        :return:
        '''
        for col in self.__bins:
            binn = self.__bins[col]
            iv = binn.get_iv()
            if min_iv <= iv<= max_iv:
                print('"{}"  :  iv={}'.format(col,iv))
                self.__bins[col].plot_woe()

def sample_split(df,y, test_size=0.3):
    """
    split dataset into train and test without over sampling
    Train_Test_Split_nooversample

    df：数据集
    y：Y变量名称
    test_size：测试集占总样本的比例
    """
    train_df,test_df = train_test_split(df,random_state=123,test_size=test_size,stratify=y)
#     target = target_name
#     Y = df[target]
#     X_df = df.drop(target, axis=1)
#     x_train, x_test, y_train, y_test = train_test_split(
#         X_df, Y, test_size=test_size, stratify=Y, random_state=123
#     )
    return train_df,test_df

def sample_split_over(df, target_name, test_size=0.3, n=1): # 过抽样，暂时保密    
    """
    split dataset into train and test with over sampling
    Train_Test_Split_oversample

    df：数据集
    target_name：Y变量名称
    test_size：测试集占总样本的比例
    n:好客户比坏客户的倍数
    """
def sample_split_summary(y_train,y_test):
    '''
    该函数为训练集和测试集划分的统计结果

    Args:

    y_train(DataFrame):训练集y数据
    y_test(DataFrame):测试集的y数据

    Returns：
    New DataFrame:训练集和测试集的划分结果
    '''
    train = ['train',len(y_train),y_train.value_counts()[0]\
             ,y_train.value_counts()[1],1.0*y_train.value_counts()[1]/len(y_train),\
           1.0*len(y_train)/(len(y_train)+len(y_test))]
    test = ['test',len(y_test),y_test.value_counts()[0]\
             ,y_test.value_counts()[1],1.0*y_test.value_counts()[1]/len(y_test),\
            1.0*len(y_test)/(len(y_train)+len(y_test))]
    sum_ = ['sum',len(y_train)+len(y_test),
            y_train.value_counts()[0]+y_test.value_counts()[0],
            y_train.value_counts()[1]+y_test.value_counts()[1],
            1.0*(y_train.value_counts()[1]+y_test.value_counts()[1])/(len(y_train)+len(y_test)),
            1.0*len(y_train)/(len(y_train)+len(y_test))+1.0*len(y_test)/(len(y_train)+len(y_test))
            ]
    return pd.DataFrame([train,test,sum_],columns=['split','total','good','bad','rate','dist'])

demo_df = pd.read_csv('data.csv').sample(n=50)
demo_df.head()
demo_df.columns.tolist()
char_var = demo_df.select_dtypes('object')
char_var.columns.tolist()
print(demo_df.shape)

# 找出id的变量
drop_list = [col for col  in demo_df.columns if ('_id' in col) or ('id_' in col) ]+['domain','primary_city','state',
                                                                                   'bank_nm','email_dur']
print(drop_list)

# 拆分数据集
train_df,test_df=sample_split(demo_df,demo_df['flgGood'],test_size=0.3)
summary_y =sample_split_summary(train_df['flgGood'],test_df['flgGood'])
# summary_y.to_excel(result_path+'/y_summary.xlsx')
print(summary_y)

# 等频分箱
y = 'flgGood'
binut = XXDBinUtils(train_df,y)
binut.auto_bin(ignore=drop_list,params={'max_bin':10,'min_sample':50})
print(binut)