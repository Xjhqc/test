# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import OrderedDict

from lib.MatrixLibData_old import MatrixLibData as SupwinData

class MatrixModelCovariancePredictor(object):
    def __init__(self, date="20171231", pool="中证800", date_len=1, output=False):
        """
        输入预测时间及股票池可输出预测结果
        :param date:  预测的时间节点,str
        :param pool: 股票池,str
        :param date_len: 计算时间长度，默认1年，int
        :param output: 是否将结果输出csv，bool
        """
        self.date = date
        self.pool = pool
        self.date_len = date_len
        self.factor_return = output
        self.data_engine = SupwinData(username="xiejie", password="matrix")
        self.all_factor = ["Supwin_Beta", "Supwin_Momentum", "Supwin_Size", "Supwin_Value", "Supwin_NLSize",
                           "Supwin_EarningYield", "Supwin_Volatility", "Supwin_Growth", "Supwin_Leverage",
                           "Supwin_Liquidity", "Supwin_Sector"]

    def get_daily_return(self, start_date=None):
        """
        :return: 返回每日因子收益、每日的回归残差及R方
        """
        date_start = start_date
        last_date_start = self.data_engine.get_trade_date(end_date=start_date,period="M")[-1]
        tradate = self.data_engine.get_trade_date(start_date=date_start, end_date=self.date,period="M")  # 获取期间所有交易日
        last_tradate = self.data_engine.get_trade_date(start_date=last_date_start, end_date=self.date,period="M")

        # 定义输出结果
        self.__Rsquare = {}
        self.__Coef = OrderedDict()
        self.__Resid = OrderedDict()
        for index, date in enumerate(tradate):
            print(date)
            last_date = last_tradate[index]  # 获取上一个交易日
            # 获取自变量：10个barra因子+1个行业因子
            X = self.data_engine.get_factor(date=last_date, config=self.all_factor, pool=self.pool)
            # 对自变量数据进行清洗
            X.dropna(inplace=True)  # 删除含有空值的样本
            X = X.reset_index(drop=True)  # 重新定义Index
            X = pd.get_dummies(X, columns=["Supwin_Sector"])  # 生成行业哑变量
            X["Country"] = 1  # 增加country取值全为1
            X.pop("ValueDate")  # 去掉Valuedate列
            # 获取因变量：下日股票收益率
            data_y = self.data_engine.get_quote(end_date=str(date), start_date=str(last_date), code=X.Code.tolist())
            data_y = data_y.groupby(by="Code")["Close"].apply(lambda x:x.values[-1]/x.values[0]-1)
            data_y = pd.DataFrame(data_y)
            data_y.reset_index(inplace=True)
            data_y.columns = ["Code","Returns"]
            data_y.dropna(inplace=True)  # 解决601313的特殊情况
            # 对风格因子和return进行市值加权
            data_capital = self.data_engine.get_factor(date=last_date, config=["Cap_Supwin", "Supwin_Sector"],
                                                       pool=self.pool)
            data_capital.dropna(inplace=True)
            Sector_Cap = data_capital[["Cap_Supwin", "Supwin_Sector"]].groupby(by="Supwin_Sector").sum()
            Sector_Cap = Sector_Cap / Sector_Cap.iloc[-1].values
            X12 = X[[u'Supwin_Sector_食品饮料']].dot(Sector_Cap[:-1].values.T)
            X12.columns = X.columns[11:38]
            X[X.columns[11:38]] = X[X.columns[11:38]] + X12
            X.pop(u'Supwin_Sector_食品饮料')
            # 匹配X,y合并为data
            data = pd.merge(X, data_y, on="Code", how="inner")
            data = pd.merge(data, data_capital, on="Code", how="inner")
            data[data.columns[1:39]] = (data[data.columns[1:39]].T * np.sqrt(data["Cap_Supwin"])).T  # X进行市值加权
            data["Returns"] = data["Returns"] * np.sqrt(data["Cap_Supwin"])  # 对Y进行市值加权
            X = data.ix[:, :39]
            code = X.pop("Code")
            y = pd.Series([round(x, 5) for x in data.Returns.values])  # 保留小数点后5位

            # 线性回归
            OLSresult = sm.OLS(y, X).fit()
            coef = [round(x, 6) for x in OLSresult.params]  # 回归系数精确到6位
            resid = [round(x, 6) for x in (y - X.dot(coef))]  # 回归残差精确到6位
            resid = resid / np.sqrt(data["Cap_Supwin"])  # 将残差还原
            self.__Coef[date] = dict(zip(X.columns.tolist(), coef))  # 生成dict
            self.__Coef[date][u'Supwin_Sector_食品饮料'] = -(
                    Sector_Cap[:-1].T.values[0] * OLSresult.params[10:-1].values).sum()
            self.__Resid[date] = dict(zip(code, resid))
            self.__Rsquare[date] = OLSresult.rsquared
        if self.factor_return:
            pd.DataFrame(self.__Coef).to_csv(u"./result/%s因子收益.csv" % self.date)
            pd.DataFrame(self.__Resid).to_csv(u"./result/%s特质收益.csv" % self.date)
            pd.Series(self.__Rsquare).to_csv(u"./result/%s回归R方.csv" % self.date)
        return pd.DataFrame(self.__Coef), pd.DataFrame(self.__Resid), pd.Series(self.__Rsquare)


class factor_return_and_special_return():
    def run(self, param):
        pool = param['pool']
        date = param['date']
        barra_engine = MatrixModelCovariancePredictor(date=date, pool=pool)
        factors_return, special_return, r_square = barra_engine.get_daily_return(start_date=date)
        factors_return.reset_index(inplace=True)
        factors_return.columns = ["Key", "Value"]
        special_return.reset_index(inplace=True)
        special_return.columns = ["Key", "Value"]
        return {'factors_return': factors_return.to_json(orient='records'),
                'special_return': special_return.to_json(orient='records')}


if __name__ == "__main__":
    MatrixModelCovariancePredictor(date="20180331",pool="中证全指",output=True).get_daily_return(start_date="20110101")
