# -*- coding: utf-8 -*-
"""
@version: python2.7
@author:ribun
@software: PyCharm Community Edition
@time: 2018/04/10 16:15

"""
from MatrixLibData import *
import pandas as pd
import numpy as np
import copy
import time as time


class Coskew():
    #---------------------------------------------------------------
    def __init__(self):
        uid = 'ribun'
        psw = 'supwin888'

        self.data_engine = MatrixLibData(uid, psw)         # 数据工具

        self.MKT_IDX_MAP = {}                              # 市场指数印射
        self.MKT_IDX_MAP['csiall'] = '000985.CSI'
        self.MKT_IDX_MAP['csi800'] = '000906.SH'
        
    #---------------------------------------------------------------
    def run(self, param):
        date = param['date']
        pool = param['pool']
        print(date, pool)
        data = self.data_engine.get_factor_temp(date=date,config=["Weight"],pool= pool,filterpool="")
        code_list = data['Code'].tolist()
        start_date = self.data_engine.trade_date_offset(date, -250)

        merge_code_list = copy.deepcopy(code_list)
        merge_code_list.append(self.MKT_IDX_MAP[pool])

        # 股票与指数的协偏度
        code_quote_data = self.data_engine.get_quote(start_date, date, merge_code_list)[['Time','Returns','Code']]
        df = code_quote_data.pivot(index='Time', columns='Code', values='Returns')

        v = df.values
        s1 = sigma = v.std(0, keepdims=True)
        means = v.mean(0, keepdims=True)
        v1 = v - means
        s2 = sigma ** 2
        v2 = v1 ** 2
        m = v.shape[0]

        skew = pd.DataFrame(v2.T.dot(v1) / s2.T.dot(s1) / m, df.columns, df.columns)
        skew *= ((m - 1) * m) ** .5 / (m - 2)

        code_coskew = skew[self.MKT_IDX_MAP[pool]].drop(self.MKT_IDX_MAP[pool])
        co_skew_df = pd.DataFrame({'Key':code_coskew.index, 'Value':code_coskew.values})

        return {'Coskew':co_skew_df.to_json(orient='records')}
        # return co_skew_df.to_json(orient='records')





#---------------------------------------------------------------
if __name__ == '__main__':
    param = {'date':'20180409', 'pool':'csi800'}
    coskew = Coskew()
    result = coskew.run(param)
    print(result)