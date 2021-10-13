import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from matplotlib.font_manager import FontProperties

class search_para:
    # 初始化
    def __init__(self, dataSet_sc):
        self.dataSet_sc = dataSet_sc

    def silh_coef(self):
        K = range(3, 9)  # 设置主题个数区间
        coef = []
        for k in K:
            km = KMeans(n_clusters=k, random_state=0).fit(self.dataSet_sc)  # 构建kmeans模型并训练
            score = silhouette_score(self.dataSet_sc, km.labels_)  # 计算对应模型的轮廓系数
            coef.append(score)
        plt.plot(K, coef)  # K为x轴输出，coef是y轴输出
        plt.xlabel('k')
        font = FontProperties(fname=r'c:\windows\fonts\msyh.ttc', size=20)
        plt.ylabel(u'轮廓系数', fontproperties=font)
        plt.title(u'轮廓系数确定最佳的K值', fontproperties=font)
        plt.show()
        k_certain = coef.index(max(coef))+3
        print('k_certain',k_certain)
        return k_certain
    def elbow(self):
        K = range(3, 9)
        mean_distortions = []
        for k in K:
            km = KMeans(n_clusters=k, random_state=0).fit(self.dataSet_sc)  # 构建kmeans模型并训练
            mean_distortions.append(
                sum(
                    np.min(
                        cdist(self.dataSet_sc, km.cluster_centers_, metric='euclidean'), axis=1))
                / self.dataSet_sc.shape[0])
        plt.plot(K, mean_distortions, 'bx-')
        plt.xlabel('k')
        font = FontProperties(fname=r'c:\windows\fonts\msyh.ttc', size=20)
        plt.ylabel(u'平均畸变程度', fontproperties=font)
        plt.title(u'用肘部法确定最佳的K值', fontproperties=font)
        plt.show()