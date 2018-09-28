# -*- coding: UTF-8 -*-
import pandas as pd
from MatrixLibData import MatrixLibData as SupwinData
import statsmodels.api as sm
import numpy as np
import time as time


class Iv_cmap(object):
    def __init__(self):
        self.data_engine = SupwinData(username="developer", password="666666")
        self.constant_pool = {"上证50": '000016.SH', "中证300": '000300.SH', "中证500": '000905.SH', "中证800": '000906.SH',
                              "中证全指": '000985.CSI'}


    def run(self, param):
        ValueDate = param['date']
        pool = param['pool']
        date_period = self.data_engine.get_trade_date(start_date='19910101', end_date=ValueDate, period='D')[-252:]
        code = self.data_engine.get_factor(date=ValueDate, config=['Supwin_Sector'], pool=pool)['Code']
        x = self.data_engine.get_quote(start_date=date_period[0], end_date=date_period[-1], code=[self.constant_pool[pool]])[['Returns','Time']]
        x = x.rename(columns={'Returns': 'Returns_x'})
        data = pd.DataFrame(index=code,columns=['Iv_cmap'])
        #残差
        for i in range(len(code)):
            # print(i)
            # while True:
            #     try:
            y = self.data_engine.get_quote(start_date=date_period[0], end_date=date_period[-1], code=[code.loc[i]])[['Returns', 'Time']]
                #     break
                # except:
                #     continue
            # y1 = y[y.Returns != 0]
            # X = pd.merge(y1, x)
            # x1 = X[['Returns_x']]
            # x1['const'] = 1.0
            # model = sm.OLS(list(y1.Returns), x1)
            # results = model.fit()
            # # print(results.params)
            # data.Iv_cmap[i] = (y1.Returns[-1:] - results.params.Returns_x * x1.Returns_x[-1:] - results.params.const) ** 2
            # # print(data.Iv_cmap[i])
            # # data.Iv_cmap[i] = results.params.const**2
            #
            # # print(data)
        # Iv_cmap_df = pd.DataFrame({'Key': data.index, 'Value': data.Iv_cmap})
        # Iv_cmap_df = Iv_cmap_df.reset_index(drop=True)
        # # print(Iv_cmap_df)
        # data.to_json("D:/Supwin/second stage/factor/Iv_cmap.json")
        # return {'Iv_cmap': Iv_cmap_df.to_json(orient='records')}


if __name__ == "__main__":
    s = time.time()
    param = {'date': '20180409', 'pool': '中证800'}
    i = 1
    j = 1
    while i<=1000:
        try:
            Iv_cmap().run(param)
            print('成功',j)
            j = j + 1
            i = i + 1
        except:
            print('获取不到')
            i = i+1
    e = time.time()
    print(e-s)