from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from getdata import getdata_local
from analyse_detial import analyse_rebuilt
from models import logon_model
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.font_manager import FontProperties
import pandas as pd


#初步特征选择
feature_first = ['dtdlcgs','jycdlcgsjd','j7tdlcgs','j14tdlcgs','j28tdlcgs','ljdlcgs',
              't1_zhmyms','t7_zhmyms', 't14_zhmyms','t28_zhmyms','lj_zhmyms',
              't1_jcxxs','t7_jcxxs','t14_jcxxs','t28_jcxxs','lj_jcxxs',
              't1_gzts','t7_gzts','t14_gzts','t28_gzts','lj_gzts',
              't1_zydzs','t7_zydzs','t14_zydzs','t28_zydzs','lj_zydzs',
              't1_dljcs','t7_dljcs','t14_dljcs','t28_dljcs','lj_dljcs']
feature_origin = ['dt_day','uuid','dtdlcgs','jycdlcgsjd','j7tdlcgs','j14tdlcgs','j28tdlcgs','ljdlcgs',
              't1_zhmyms','t7_zhmyms', 't14_zhmyms','t28_zhmyms','lj_zhmyms',
              't1_jcxxs','t7_jcxxs','t14_jcxxs','t28_jcxxs','lj_jcxxs',
              't1_gzts','t7_gzts','t14_gzts','t28_gzts','lj_gzts',
              't1_zydzs','t7_zydzs','t14_zydzs','t28_zydzs','lj_zydzs',
              't1_dljcs','t7_dljcs','t14_dljcs','t28_dljcs','lj_dljcs']
col_dummy = ['jycdlcgsjd']

class var_fea:
    # 初始化
    def __init__(self, data,feature_first,feature_origin):
        self.data  = data
        self.feature_first = feature_first
        self.feature_origin = feature_origin

    def first_select(self):
        data_train = self.data[self.feature_first]
        data_origin = self.data[self.feature_origin]
        return data_train,data_origin
    def dummy(self,data_train,col_dummy):
        data_encoded = pd.concat([data_train.drop(columns=col_dummy, axis=1),pd.get_dummies(data_train, columns=col_dummy)],axis=1)
        # data_encoded.to_csv(r"登录行为聚类\kankan2.csv")
        return data_encoded
    def standard(self,data_train):
        dataSet = data_train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        dataSet_sc = (dataSet - dataSet.min()) / (dataSet.max() - dataSet.min())
        dataSet_sc = dataSet_sc.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        col_name = dataSet_sc.columns.values.tolist()
        return dataSet_sc,col_name
    def var_thre(self,dataSet_sc,k):
        var = dataSet_sc.var()
        step = (var.max() - 0) / k
        thre = []
        a = 0
        while len(thre) <= k-1:
            thre.append(a)
            a = a + step
        return thre
    def second_select(self,thre,dataSet_sc,data_origin,n_c,r_s,m_i):
        coef = []
        for a in thre:
            sel = VarianceThreshold(threshold=a)  # 剔除方差大于阈值的特征
            new_data = sel.fit_transform(dataSet_sc)  # 返回的结果为选择的特征矩阵
            # print(new_data.shape)
            km = KMeans(n_clusters=n_c, random_state= r_s,max_iter=m_i).fit(dataSet_sc)
            # logon = logon_model(dataSet_sc,data_origin, n_clusters,random_state,max_iter)
            # km = logon.train_model() #构建kmeans模型并训练
            score = silhouette_score(new_data, km.labels_) # 计算对应模型的轮廓系数
            coef.append(score)
        plt.plot(thre,coef) # K为x轴输出，coef是y轴输出
        plt.xlabel('a')
        font = FontProperties(fname=r'c:\windows\fonts\msyh.ttc', size=20)
        plt.ylabel(u'轮廓系数', fontproperties=font)
        plt.title(u'轮廓系数确定最佳的a值', fontproperties=font)
        plt.show()
        return coef
    def second_result(self,a,dataSet_sc,data_origin,col_name):
        sel = VarianceThreshold(threshold=a)  # 剔除方差大于阈值的特征
        temp_data = sel.fit_transform(dataSet_sc.iloc[:, :].values)  # 返回的结果为选择的特征矩阵
        fea_stay = sel.get_support(indices=False)
        # 形成新的new_data,dataframe
        drop_fea = []
        for i in range(len(fea_stay)):
            if fea_stay[i] == False:
                drop_fea.append(col_name[i])
        new_dataSet_sc = dataSet_sc.drop(drop_fea, axis=1)
        new_data_origin = data_origin.drop( drop_fea,axis=1)
        return new_dataSet_sc,new_data_origin

