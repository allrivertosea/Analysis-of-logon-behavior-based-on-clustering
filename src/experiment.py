from models import logon_model
from feature import var_fea,feature_first,feature_origin,col_dummy
from getdata import *
from search import *
from analyse_detial import *
from sklearn.externals import joblib

def experiment(data,n):
    #特征选择第一次, 独热编码，标准化
    # print(data)
    feature_sel = var_fea(data,feature_first,feature_origin)
    data_train,data_origin = feature_sel.first_select()
    data_train_encoded = feature_sel.dummy(data_train,col_dummy)
    dataSet_sc,col_name = feature_sel.standard(data_train_encoded)
    # #参数选择：聚类簇数k，第一次k=3,第二次3类，第三次5类
    search_pa = search_para(dataSet_sc)
    k = search_pa.silh_coef()
    print('簇数',k)
    # #特征选择第二次,必要情况下使用，否则不进行
    thre = feature_sel.var_thre(dataSet_sc,10)
    coef = feature_sel.second_select(thre,dataSet_sc,data_origin,k,0,1000)
    #当不进行二次特征选择时，a=0
    a = 0
    new_dataSet_sc,new_data_origin = feature_sel.second_result(a,dataSet_sc,data_origin,col_name)
    #模型训练
    dataSet_train = new_dataSet_sc
    dataSet_orign = new_data_origin
    logon = logon_model(dataSet_train,data_origin,k,0,1000)
    model = logon.train_model()
    r1,r_simple,r_sc,r_origin = logon.output_results(model)
    print(r_simple)
    n=n+1
    print('聚类次数:%d'%n)
    joblib.dump(model, r'F:\项目合集\登录行为聚类分析\model_load\cls_model_params%d.pkl'%n)
    if max(r1.tolist())/min(r1.tolist())>2:
        analy = analyse_rebuilt(r_origin,n)
        cluster_outlier, data_next,mean_cluser = analy.result(k,r1)
        mean_cluster_std = mean_cluser.describe().loc[['std']]
        sorted_mean_cluser =mean_cluster_std.sort_values(by=mean_cluster_std.index.tolist(),axis=1)
        sorted_mean_cluser.drop(columns='聚类类别',axis=1).to_csv(r'F:\项目合集\登录行为聚类分析\各次聚类训练集\std_compare%d.csv' % (n))
        # print(sorted_mean_cluser)
        experiment(data_next,n)

if __name__=="__main__":
    # 获取数据，经过不断的聚类，训练集也是不断变化的，可存储为中间表，每一次可能都需要做特征选择
    localpath = r"训练集.csv"
    data = getdata_local(localpath)
    n = 0  # 聚类次数，初始化为1
    experiment(data,n)






