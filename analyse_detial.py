from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
tool = MinMaxScaler(feature_range=(0, 1)) #根据需要设置最大最小值，这里设置最大值为1.最小值为0


#用TSNE进行数据降维并展示聚类结果
class visual_tsne:
    # 初始化
    def __init__(self, data_sc,r_sc):
        self.data_train  = data_sc
        self.r_sc = r_sc
    def result(self):
        tsne = TSNE()
        tsne.fit_transform(self.data_train) #进行数据降维,并返回结果
        tsne = pd.DataFrame(tsne.embedding_, index = self.data_train.index) #转换数据格式
        plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
        #不同类别用不同颜色和样式绘图
        d = tsne[self.r_sc[u'聚类类别'] == 0]     #找出聚类类别为0的数据对应的降维结果
        plt.plot(d[0], d[1], 'r.')
        d = tsne[self.r_sc[u'聚类类别'] == 1]
        plt.plot(d[0], d[1], 'go')
        d = tsne[self.r_sc[u'聚类类别'] == 2]
        plt.plot(d[0], d[1], 'b*')
        d = tsne[self.r_sc[u'聚类类别'] == 3]
        plt.plot(d[0], d[1], 'y*')
        d = tsne[self.r_sc[u'聚类类别'] == 4]
        plt.plot(d[0], d[1], 'c')
        d = tsne[self.r_sc[u'聚类类别'] == 5]
        plt.plot(d[0], d[1], 'm')
        plt.show()

class analyse_rebuilt:
    # 初始化
    def __init__(self, r_origin,n):
        self.data_temp  = r_origin
        self.n_times = n

    def result(self,k,r1):

        labels = r1.tolist()
        cluster_num = 1
        for i in range(k):
            label_index = r1[r1.values == labels[i]].index[0]
            if labels[i] == min(labels):
                cluster_outlier = self.data_temp.loc[self.data_temp['聚类类别'] == label_index]
                # cluster_outlier.to_csv(r'F:\datamiad\登录行为聚类\各簇数据\%dcluster%d_outlier_an.csv' % (self.n_times, i))
                # cluster_outlier.describe().to_csv(r'F:\datamiad\登录行为聚类\分析数据\%dcluster%d_outlier_an.csv'%(self.n_times,i))
                mean_outlier = cluster_outlier.describe().iloc[[1]]
            else:
                cluster = self.data_temp.loc[self.data_temp['聚类类别'] == label_index]
                # cluster.to_csv(r'F:\datamiad\登录行为聚类\各簇数据\%dcluster%d_an.csv'% (self.n_times,i))
                # cluster.describe().to_csv(r'F:\datamiad\登录行为聚类\分析数据\%dcluster%d_an.csv'%(self.n_times,i))

                if cluster_num==1:
                    data_next = cluster.drop([ '聚类类别'], axis=1)
                    mean_cluser = cluster.describe().iloc[[1]]
                    cluster_num = cluster_num+1
                else:
                    data_next = pd.concat([data_next,cluster.drop([ '聚类类别'], axis=1)])
                    mean_cluser = pd.concat([mean_cluser, cluster.describe().iloc[[1]]])
        mean_cluser = pd.concat([mean_cluser, mean_outlier])
        # mean_cluser.to_csv(r'F:\datamiad\登录行为聚类\各次聚类训练集\特征均值%d_an.csv' % (self.n_times))
        mean_cluser_new = (mean_cluser - mean_cluser.min()) / (mean_cluser.max() - mean_cluser.min())
        print(mean_cluser_new)
        # data_next.to_csv(r'F:\datamiad\登录行为聚类\各次聚类训练集\训练集%d_an.csv' % (self.n_times))
        return cluster_outlier,data_next,mean_cluser_new








