# -*- coding: UTF-8 -*-

import pandas as pd
from dateutil.parser import parse
from elasticsearch import Elasticsearch
import itertools
import time
from collections import OrderedDict
es = Elasticsearch(hosts = ["http://192.168.134.191:9232"])

# get方法可以获取单个因子
# 参数id由三个参数拼接，pool-date-factor，其中factor都是小写
# 参数index/doc_type都是cnfactor
# data = es.get(index="cnfactor",doc_type="cnfactor",id="csiall-20180403-supwin_beta")   # 获取因子
# print(pd.DataFrame(data['_source']['items']))


# 获取行情数据
startdate = "2018-01-15"
enddate = "2018-04-15"
code = ["000001.SZ","000002.SZ"]
code = [c.lower() for c in code]
tradate = es.get(index="calendar", doc_type="calendar", id="cncalendar")['_source']['items']
tradate = [d for d in tradate if d>=startdate and d<= enddate]
tramonth = list(set([parse(d).strftime("%Y%m") for d in tradate]))
month_code = []
for i in itertools.product(tramonth,code):
    month_code.append("-".join(list(i)))
body = {
    "query": {
        "ids": {
            "type": "cnquote",
            "values": month_code
        }
    }
}
data = es.search(index="cnquote",doc_type="cnquote",body=body,params={"size": 10000})['hits']['hits']
print(len(data))
res = []
for dat in data:    # 对数据格式进行调整
    res = res+dat['_source']['items']
data = pd.DataFrame(res)
# print(data)

# search方法可以获取多个因子
# 将所有因子以List放进body里面的values中
# 其余与get方法一样

# date = "20180403"   # 日期
# pool = "csi800" # 股票池
# gtja_no_data = [9, 10, 30, 68, 111, 143, 144, 154, 162, 164, 166,186]
# factor=["GTJA_Alpha"+str(i) for i in range(1,192) if i not in gtja_no_data]
# factor.extend(["Alpha_"+str(i) for i in range(1,102) if i not in [7]])  # 因子名称
#
# factors = [i.lower() for i in factor]
# factors_map = dict(zip(factors,factor))
# values = [pool+"-"+date+"-"+i for i in factors]
#
# body = {
#     "query": {
#         "ids": {
#             "type": "cnfactor",
#             "values": values    # ["csiall-20180403-gtja_alpha1","csiall-20180403-gtja_alpha2"]
#         }
#     }
# }
# s = time.time()
# data = es.search(index="cnfactor",doc_type="cnfactor",body=body,params={"size":1000})['hits']['hits']
# e = time.time()
# print(e-s)
# res = OrderedDict()
# for dat in data:
#     temp = {}
#     var = dat['_id'].split("-")[-1]
#     for val in dat['_source']['items']:
#         temp[val['key']] = val['value']
#     res[var] = temp
# data = pd.DataFrame(res)
# col = [factors_map[c] for c in data.columns]
# data.columns = col
# data.index.name = "Code"
# data.reset_index(inplace=True)
# data["ValueDate"] = date
