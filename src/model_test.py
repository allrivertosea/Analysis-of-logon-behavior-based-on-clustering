from sklearn.externals import joblib
import os
from scipy.spatial import distance
import numpy as np


filePath = 'model_load'#模型保存路径
filename = []
for i,j,k in os.walk(filePath):
    for s in k:
        filename.append(s)
print(filename)
#登录行为异常检测
def detect_abn(data):
    pred = []
    for i in range(len(filename)):
        logon_cluster = joblib.load(r'model_load\cls_model_params%d.pkl'%i)
        centroids = logon_cluster.cluster_centers_
        xTest = data
        # Compute predictions for each of the test examples
        pred.append(predictClustering(centroids, xTest, "euclidean"))
    return pred
def predictClustering(clusters,xTest,metric):
    clustLabels = []
    simFunction = getDistLambda(metric)
    for x in range(len(xTest)):
        clustDex = -1
        clustDist = float('inf')
        for y in range(len(clusters)):
            dist = simFunction(clusters[y],xTest[x])
            if (dist < clustDist):
                clustDist = dist
                clustDex = y
        clustLabels.append(clustDex)
    predict = clustLabels
    return predict

def getDistLambda(metric):
    if (metric == "manhattan"):
        return lambda x, y: distance.cityblock(x, y)
    elif (metric == "cosine"):
        return lambda x, y: distance.cosine(x, y)
    else:
        return lambda x, y: distance.euclidean(x, y)
