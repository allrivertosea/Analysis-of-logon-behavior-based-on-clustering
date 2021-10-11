# import adtools
import pandas as pd

dataBaseName = "atsecdb"
userName = "postgres"
password = "SfuK39iSDLF00xkjdef"
host = "10.255.49.17"
port ="5432"
cmd = "select * from featuretable_accountbehavior_login;"


# def getdata_sql():
#     pg = adtools.PostGreSQL(dataBaseName,userName,password,host,port)
#     down_data = adtools.GetData(cmd)
#     data = down_data.down_data(pg) #读取数据库数据，返回训练集的dataframe
#     # local_data = data.to_csv(downpath)  # 可选择将训练集下载到本地
#     return data
def getdata_local(path):
    data = pd.read_csv(path)
    return data








