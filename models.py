#包括各种安全场景的模型，接收处理好的训练集的输入，输出结果
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

class logon_model:
    # 初始化
    def __init__(self, data_sc, data_origin,n_clusters,random_state,max_iter):
        self.data_train  = data_sc
        self.data_origin = data_origin
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
    def train_model(self):
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,max_iter=self.max_iter).fit(self.data_train.iloc[:, :].values)
        return model
    def output_results(self,model):
        r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
        r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
        r_simple = pd.concat([r2, r1], axis=1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
        # 详细输出原始数据及其类别:分为标准化后的输出和原始数据的输出
        r_sc = pd.concat([self.data_train, pd.Series(model.labels_, index=self.data_train.index)], axis=1)  # 详细输出每个样本对应的类别
        r_sc.columns = list(self.data_train.columns) + [u'聚类类别']  # 重命名表头
        # r.to_csv(r'F:\安天数据分析EDR\data-mining-and-analysis-program\登录行为聚类\聚类结果.csv') #保存结果
        r_origin = pd.concat([self.data_origin, pd.Series(model.labels_, index=self.data_origin.index)],
                             axis=1)  # 详细输出每个样本对应的类别
        r_origin.columns = list(self.data_origin.columns) + [u'聚类类别']  # 重命名表头
        return r1,r_simple,r_sc,r_origin

    def down_result(self,r_origin,path):
        r_origin.to_csv(path)  # 保存结果