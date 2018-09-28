# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import time as time
from collections import OrderedDict
import sys
from dateutil.parser import parse

# reload(sys)
# sys.setdefaultencoding("utf-8")

from lib.MatrixLibData import MatrixLibData as SupwinData

global freq,Model_Period
freq = {"M": 21, "W": 5, "D":1}
Model_Period = {"Short":252,"Long":504}



class MatrixModelCovariance(object):
    """
    因子协方差阵的预测包括：NW调整、特征根调整、波动率调整
    """

    def __init__(self, data, v_half=90.0, c_half=480.0):
        self.data = data.T
        self.v_half = v_half
        self.c_half = c_half

    def cov_weight(self, D=0, D_v=2):
        """
        计算指数加权协方差、指数加权方差、滞后N期的指数加权协方差、滞后M期的指数加权方差
        :param D:协方差的滞后期，等于0时意味着不滞后
        :param D_v:方差的滞后期
        :return:
        """
        lambda_v = 0.5 ** (1 / self.v_half)  # 方差的加权底数lambda
        lambda_c = 0.5 ** (1 / self.c_half)  # 协方差的加权底数lambda
        if D == 0:
            weight_cov1 = pd.DataFrame(0, columns=self.data.columns, index=self.data.columns)  # 两个空白df用于接收结果
            weight_var1 = pd.DataFrame(0, columns=self.data.columns, index=self.data.columns)
            d_len = len(self.data)
            lambda_v = lambda_v ** np.arange(d_len)[::-1]  # 生成方差的指数权重序列
            lambda_c = lambda_c ** np.arange(d_len)[::-1]  # 生成协方差的指数权重序列
            for i in range(len(self.data.columns)):
                fi = self.data.ix[:, i]
                for j in range(i + 1, len(self.data.columns)):
                    fj = self.data.ix[:, j]
                    weight_cov1.ix[i, j] = weight_cov1.ix[j, i] = \
                        np.sum((fi - fi.mean()) * (fj - fj.mean()) * lambda_c) / np.sum(lambda_c)  # 计算指数加权协方差
                weight_cov1.ix[i, i] = np.sum((fi - fi.mean()) ** 2 * lambda_c) / np.sum(lambda_c)
                weight_var1.ix[i, i] = np.sum((fi - fi.mean()) ** 2 * lambda_v) / np.sum(lambda_v)  # 计算指数加权方差
            print (u"原始协方差计算完毕")
            return weight_cov1, weight_var1
        else:
            # 当有滞后期(D_v,D)时需要计算三个df
            # 三个空白df用于接受结果，cov_plus,cov_minus,var
            weight_cov_plus = weight_cov_minus = pd.DataFrame(0, columns=self.data.columns, index=self.data.columns)
            weight_var = pd.DataFrame(0, columns=self.data.columns, index=self.data.columns)
            lambda_v = lambda_v ** np.arange(len(self.data) - D_v)[::-1]  # 生成方差的指数权重序列
            lambda_c = lambda_c ** np.arange(len(self.data) - D)[::-1]  # 生成协方差的指数权重序列
            for i in range(len(self.data.columns)):
                fi = self.data.ix[:, i]
                fi_plus = self.data.ix[:-D, i]  # 第i个因子近日截断用于计算协方差
                fi_minus = self.data.ix[D:, i]  # 第i个因子远日截断用于计算协方差
                fi_v_plus = self.data.ix[D_v:, i]  # 远日截断用于计算方差
                fi_v_minus = self.data.ix[:-D_v, i]  # 近日截断用于计算方差
                for j in range(i + 1, len(self.data.columns)):
                    fj = self.data.ix[:, j]  # 第j个因子
                    fj_plus = self.data.ix[D:, j]  # 第j个因子近日截断用于计算协方差
                    fj_minus = self.data.ix[:-D, j]  # 第j个因子远日截断用于计算协方差
                    weight_cov_plus.ix[i, j] = weight_cov_plus.ix[j, i] = np.sum(
                        (fi_plus - fi_plus.mean()).values * (fj_plus - fj_plus.mean()).values * lambda_c) / np.sum(
                        lambda_c)  # 计算cov_plus
                    weight_cov_minus.ix[i, j] = weight_cov_minus.ix[j, i] = np.sum(
                        (fi_minus - fi_minus.mean()).values * (fj_minus - fj_minus.mean()).values * lambda_c) / np.sum(
                        lambda_c)  # 计算cov_minus
                weight_cov_plus.ix[i, i] = weight_cov_minus.ix[i, i] = np.sum(
                    (fi_plus - fi_plus.mean()).values * (fi_minus - fi_minus.mean()).values * lambda_c) / np.sum(
                    lambda_c)  # 计算cov_plus/cov_minus对角线上的值
                weight_var.ix[i, i] = np.sum(
                    (fi_v_plus - fi_v_plus.mean()).values * (
                            fi_v_minus - fi_v_minus.mean()).values * lambda_v) / np.sum(
                    lambda_v)  # 计算方差的滞后指数加权结果
            return weight_cov_plus, weight_cov_minus, weight_var

    def Newy_West_adjust(self, Lag_cov=5, Lag_var=2, NW_adjust=False):
        """
        对协方差进行NW调整
        :param Lag_cov: 协方差的滞后期，int
        :param Lag_var: 方差的滞后期，int
        :return:协方差调整结果，DataFrame
        """
        cov_init, var_init = self.cov_weight()
        # 进行NW调整，得到cov_new及var_new
        if NW_adjust:
            for i in range(1, Lag_cov + 1):
                weight_cov_plus, weight_cov_minus, weight_var = self.cov_weight(D=i, D_v=i)

                # 由协方差阵计算得到相关系数阵corr_new
                cov_var = pd.DataFrame(np.sqrt(np.diag(weight_cov_plus)))
                cov_var = cov_var.dot(cov_var.T)  # 协方差对角线的标准差相乘得到38*38的标准差阵
                cov_var.columns, cov_var.index = weight_cov_plus.columns, weight_cov_plus.index
                corr_new = weight_cov_plus / cov_var
                print ((corr_new > 1).sum())

                cov_init = cov_init + (weight_cov_plus + weight_cov_minus) * (1 - i / (Lag_cov + 1))
                if i <= Lag_var:
                    var_init = var_init + 2 * weight_var * (1 - i / (Lag_cov + 1))  # 这里可能需要乘以2
        cov_new = freq[FQ] * cov_init
        var_new = freq[FQ] * var_init

        # 由协方差阵计算得到相关系数阵corr_new
        cov_var = pd.DataFrame(np.sqrt(np.diag(cov_new)))
        cov_var = cov_var.dot(cov_var.T)  # 协方差对角线的标准差相乘得到38*38的标准差阵
        cov_var.columns, cov_var.index = cov_new.columns, cov_new.index
        corr_new = cov_new / cov_var

        # 由相关系数阵与方差阵计算得到新的协方差阵
        var = pd.DataFrame(np.sqrt(np.diag(var_new)))
        var = var.dot(var.T)
        var.columns, var.index = cov_new.columns, cov_new.index
        print (u"Newy_West调整完毕")
        return corr_new * var

    def Eigen_adjust(self):
        """
        :return:返回特征根调整的结果
        """
        cov_new = self.Newy_West_adjust()
        col = cov_new.columns
        # cov_new = np.array(cov_new, dtype=float)
        D_NW, U_NW = np.linalg.eig(cov_new)  # 进行特征值分解
        pi = np.repeat(0, len(self.data.T))  # 调整系数Pi
        for m in range(300):  # 进行300次MC模拟
            MCF = pd.DataFrame()
            for index, factor in enumerate(self.data.columns):
                MCF[factor] = np.random.normal(0, np.sqrt(D_NW[index]), len(self.data))  # 生成正态分布模拟数值的矩阵
            f_m = pd.DataFrame(U_NW.dot(MCF.T))  # 模拟矩阵乘以特征向量得到“真实因子收益”
            f_m.index, f_m.columns = self.data.columns, self.data.index
            F_m_MC = np.cov(f_m, ddof=1)  # 计算“真实因子收益”的协方差阵
            D_m, U_m = np.linalg.eig(F_m_MC)  # 再次进行特征值分解
            D_m_hat = np.diag(U_m.T.dot(cov_new).dot(U_m))
            pi = pi + D_m_hat / D_m
        pi = np.sqrt(pi / 300)  # pi还有有差别,并不是围绕着1.08浮动，可能也是时间太短导致的
        cov_eigen = U_NW.dot(np.diag(pi ** 2) * D_NW).dot(U_NW.T)
        print (u"特征根调整完毕")
        return pd.DataFrame(cov_eigen, columns=col, index=col)

    def Vol_adjust(self, w_half=42.0):
        """
        :param w_half:半衰期参数,int
        :return: 返回波动率偏误调整的结果
        """
        delta_w = (0.5 ** (1 / w_half)) ** np.arange(len(self.data))[::-1]  # 这样会使得系数远大于，这里可能要做归一化
        fac_adj = self.data.dot(np.diag(1 / self.data.std(ddof=1)))
        B_Ft = np.sqrt((fac_adj ** 2).sum(axis=1) / len(self.data.T))
        lambda_F = np.sqrt(np.sum(delta_w * B_Ft) / np.sum(delta_w))  # 时间太短会导致lambda较大，主要是因为时间长度不够标准差太小
        cov_eigen = self.Eigen_adjust()
        cov_VRA = lambda_F ** 2 * cov_eigen
        print (u"波动率调整完毕。")
        return pd.DataFrame(cov_VRA, columns=self.data.columns, index=self.data.columns)


class MatrixModelSpecificRisk(object):
    """
    特殊风险调整包括：NW调整、结构化模型调整、贝叶斯收缩、波动率调整
    """

    def __init__(self, data, sr_half=90.0):
        self.data = data.T
        self.sr_half = sr_half
        self.all_factor = ["Supwin_Beta", "Supwin_Momentum", "Supwin_Size", "Supwin_Value", "Supwin_NLSize",
                           "Supwin_EarningYield", "Supwin_Volatility", "Supwin_Growth", "Supwin_Leverage",
                           "Supwin_Liquidity", "Supwin_Sector"]

        self.data_engine = SupwinData(username="xiejie", password="matrix")

    def var_weight(self, D=0):
        """
        计算指数加权方差、滞后D期的指数加权方差
        :param D:方差的滞后期，int
        :return:返回指数加权方差，Series
        """
        delta_sr = 0.5 ** (1 / self.sr_half)
        if D == 0:
            weight_var = pd.DataFrame(columns=self.data.columns)
            d_len = len(self.data)
            delta_sr = delta_sr ** np.arange(d_len)[::-1]
            for code in self.data.columns:
                fi = self.data[code]
                weight_var[code] = [np.sum((fi - fi.mean()) ** 2 * delta_sr) / np.sum(delta_sr)]
            return weight_var
        else:
            weight_var = pd.DataFrame(columns=self.data.columns)
            d_len = len(self.data) - D
            delta_sr = delta_sr ** np.arange(d_len)[::-1]
            for code in self.data.columns:
                fk = self.data[code].values
                f1 = self.data[code][:-D].values
                f2 = self.data[code][D:].values
                weight_var[code] = [np.sum((f1 - fk.mean()) * (f2 - fk.mean()) * delta_sr) / np.sum(delta_sr)]
            return weight_var

    def Newy_West_adjust(self, Lag_var=2, NW_adjust=False):
        """
        :param Lag_var: NW调整的之后期数，int
        :param NW_adjust: 是否进行NW调整
        :return:返回NW调整的结果（默认不进行调整）,Series
        """
        var_init = self.var_weight()
        if NW_adjust:
            for i in range(1, Lag_var + 1):
                var_adjust = self.var_weight(D=i)
                var_init = var_init + var_adjust * (1 - i / (Lag_var + 1))
        var_TS = freq[FQ] * var_init
        print (u"Newy_West调整完毕")
        return pd.Series(np.array(var_TS)[0], index=var_TS.columns)

    def Structural_Model_adjust(self):
        """
        将特质波动率按中间变量gamma分为两组进行结构化模型调整
        :return: 返回结构化调整结果，Series
        """
        var_TS = self.Newy_West_adjust()
        var_eq = self.data.var()
        var_u = 1 / 1.35 * (var_eq.quantile(.75) - var_eq.quantile(.25))
        z_u = ((var_eq - var_u) / var_u).abs()
        # 中间变量gamma_n的计算
        gamma_n = np.minimum(1, np.maximum(0, (len(self.data) - 60) / 120.0)) * np.minimum(1, np.exp(1 - z_u))
        Code_nor = gamma_n.index[gamma_n == 1]
        Code_nnor = gamma_n.index[gamma_n != 1]

        # regression
        date = self.data.index[-2]  # 获取同日期
        barra_factor = self.all_factor[:-1]
        X = self.data_engine.get_factor(date=date, config=self.all_factor, pool=Pool)  # 获取所有因子数据
        X.dropna(inplace=True)  # 删除含有空值的样本
        X = X.reset_index(drop=True)  # 重新定义Index
        X = pd.get_dummies(X, columns=["Supwin_Sector"])  # 生成行业哑变量
        X["Country"] = 1  # 增加国家变量取值全为1
        X.pop("ValueDate")  # 去掉Valuedate变量
        X = X.set_index(keys="Code", drop=True)
        # 线性回归
        y, X1 = np.log(var_TS.loc[Code_nor]), X.loc[Code_nor]  # 对数波动率做因变量
        OLSresult = sm.OLS(y, X1).fit()  # 进行线性回归
        coef = OLSresult.params
        resid = y.values - X1.dot(coef).values  # 回归得到的残差
        var_STR = resid.mean() * np.exp(X.loc[gamma_n.index].dot(coef))
        print (u"Structural_Model调整完毕")
        return gamma_n * var_TS + (1 - gamma_n) * var_STR

    def Bayesian_Shrinkage(self, q=1):
        """
        将特质波动率分10组分别进行贝叶斯收缩
        :param q: 收缩参数，int
        :return: 返回贝叶斯收缩结果，Series
        """
        var_hat = self.Structural_Model_adjust()
        var_hat = pd.DataFrame(var_hat)
        var_hat["Code"] = var_hat.index
        var_hat.columns = ["var", "Code"]
        var_hat.sort_values(by="var", inplace=True)  # 按波动率排序
        var_hat.reset_index(drop=True, inplace=True)
        group_num = len(var_hat) / 10  # 分成10组每组个数
        res = pd.DataFrame()
        for i in range(10):
            vh_temp = var_hat.loc[(var_hat.index < (i + 1) * group_num) * (var_hat.index >= i * group_num)]
            date = self.data.index[-1]
            d_cap = self.data_engine.get_factor(date=date, config=["Cap_Supwin"], pool="中证全指")
            d_cap["Cap_Supwin"] = d_cap["Cap_Supwin"].astype("float")
            d_mer = pd.merge(vh_temp, d_cap[["Code", "Cap_Supwin"]], on="Code", how="left").fillna(0)
            d_mer["Cap_Supwin"] = d_mer["Cap_Supwin"] / d_mer["Cap_Supwin"].sum()  # 将市值进行归一化作为权重
            var_fa = (d_mer["var"] * d_mer["Cap_Supwin"]).sum()  # 组内波动率均值为市值加权均值
            delta_sn = np.sqrt(((d_mer["var"] - var_fa) ** 2).mean())  # 中间变量delta_sn，市值加权标准化
            vn = 1 / (delta_sn / (q * np.abs(d_mer["var"] - var_fa)) + 1)  # 将公式倒写，减少重复
            d_mer["var"] = vn * var_fa + (1 - vn) * d_mer["var"]  # 组内调整后的结果
            res = pd.concat([res, d_mer[["Code", "var"]]])
        result = pd.Series(res["var"])
        result.index = res["Code"].values
        print (u"Bayesian调整完毕")
        return result

    def Vol_adjust(self, w_half=42.0):
        """
        对特质波动率进行波动率偏误调整
        :param w_half: VOL调整的半衰期参数
        :return: 返回特质波动率偏误调整的结果，Series
        """
        delta_w = (0.5 ** (1 / w_half)) ** np.arange(len(self.data))[::-1]
        fac_adj = self.data.dot(np.diag(1 / self.data.std(ddof=1)))
        B_Ft = np.sqrt((fac_adj ** 2).sum(axis=1) / np.shape(self.data)[1])
        lambda_F = np.sqrt(np.sum(delta_w * B_Ft) / np.sum(delta_w))  # 中间变量lambda
        var_BS = self.Bayesian_Shrinkage()
        var_VRA = lambda_F ** 2 * var_BS
        print (u"波动率调整完毕。")
        return var_VRA


class MatrixModelCovariancePredictor(object):
    def __init__(self,date = "20171231", pool = "中证800", date_len=1, output=False):
        """
        输入预测时间及股票池可输出预测结果
        :param date:  预测的时间截点,str
        :param pool: 股票池,str
        :param date_len: 计算时间长度，默认1年，int
        :param output: 是否将结果输出csv，bool
        """
        self.date = date
        self.pool = pool
        self.date_len = date_len
        self.factor_return = output
        self.data_engine = SupwinData(username="xiejie", password="matrix")

    def get_daily_return(self,start_date = None):
        """
        获取每日因子收益
        :param start_date:
        :return:
        """
        if start_date:
            date_start = start_date
            print ("将获取"+start_date+"到"+self.date+"的因子收益及特质收益")
        else:
            print ("请输入start_date!")
        tradate = self.data_engine.get_trade_date(start_date=date_start, end_date=self.date)  # 获取期间所有交易日
        pool_mapping = {"中证800":"csi800","中证全指":"csiall"}

        # 定义输出结果
        self.__Coef = OrderedDict()
        self.__Resid = OrderedDict()
        for date in tradate:
            factors_return = self.data_engine.get_factor_temp(date=date, pool=pool_mapping[self.pool], config=["factors_return"])
            special_return = self.data_engine.get_factor_temp(date=date, pool=pool_mapping[self.pool], config=["special_return"])
            self.__Coef[date] = dict(zip(factors_return.Code.values, factors_return.factors_return.values))
            self.__Resid[date] = dict(zip(special_return.Code.values, special_return.special_return.values))
        if self.factor_return:
            pd.DataFrame(self.__Coef).to_csv(u"./result/%s因子收益.csv" % self.date)
            pd.DataFrame(self.__Resid).to_csv(u"./result/%s特质收益.csv" % self.date)
        return pd.DataFrame(self.__Coef),pd.DataFrame(self.__Resid)

    def covariance_predict(self, start_date = None,FactorCovariance=True, SpecificRisk=True, output=False, frequency="M", Period = "Short"):
        """
        :param start_date:
        :param FactorCovariance:是否预测因子协方差阵,bool
        :param SpecificRisk: 是否预测特质风险,bool
        :param output: 是否将结果输出成csv，bool
        :param frequency:
        :return:
        """
        global FQ, Pool
        Pool = self.pool
        FQ = frequency

        if not start_date:
            start_date = self.date
        cov_pre_dat = self.data_engine.get_trade_date(start_date=start_date,end_date=self.date,period=frequency)    # 协方差的起始
        fac_ret_start = self.data_engine.trade_date_offset(date=cov_pre_dat[0],offset=-Model_Period[Period])    # 因子收益的起始
        fac_cov , spe_risk = self.get_daily_return(start_date=fac_ret_start) # 获取历史所有的因子收益
        for index,cov_date in enumerate(cov_pre_dat):
            if FactorCovariance:
                stock_cov = fac_cov.loc[:,:cov_date].ix[:,-Model_Period[Period]:]
                factor_covariance = MatrixModelCovariance(stock_cov).Vol_adjust()
                if output:
                    factor_covariance.to_csv(u"./CovariancePredict%s/%s因子协方差预测矩阵.csv" % (Period,cov_date))
            if SpecificRisk:
                stock_resid = spe_risk.loc[:,:cov_date].ix[:,-Model_Period[Period]:]
                stock_resid = stock_resid.ix[stock_resid.ix[:, -1].dropna().index]  # 以评价日为基准
                stock_resid.to_csv("temp3.csv")
                specific_risk = MatrixModelSpecificRisk(stock_resid).Vol_adjust()
                if output:
                    specific_risk.to_csv(u"./CovariancePredict%s/%s特殊风险预测值.csv" % (Period,cov_date))
        factor_risk = {}
        for i in factor_covariance.index:
            for j in factor_covariance.columns:
                    factor_risk[i+"$"+j] = factor_covariance.loc[i,j]
        return pd.DataFrame(pd.Series(factor_risk),columns=[self.date]),pd.DataFrame(specific_risk.values,index=specific_risk.index,columns=[self.date])

class CalCulate():
    def get_factor_return_and_special_return(self, date, pool):
        barra_engine = MatrixModelCovariancePredictor(date=date, pool=pool)
        factors_return, special_return = barra_engine.get_daily_return(start_date=date)
        return factors_return, special_return

    def get_FactorRisk_and_SpecialRisk_Short_W(self, param):
        config = ["Risk_Short_W"]
        params = config[0].split("_")
        pool = param['pool']
        date = parse(param['date']).strftime("%Y-%m-%d")
        barra_engine = MatrixModelCovariancePredictor(date=date,pool=pool)
        FactorRisk, SpecialRisk = barra_engine.covariance_predict(start_date=date,frequency=params[2],Period=params[1])
        return {'factor_risk_short_w':FactorRisk.to_json(),'special_risk_short_w':SpecialRisk.to_json()}

    def get_FactorRisk_and_SpecialRisk_Short_M(self, param):
        config = ["Risk_Short_M"]
        params = config[0].split("_")
        pool = param['pool']
        date = parse(param['date']).strftime("%Y-%m-%d")
        barra_engine = MatrixModelCovariancePredictor(date=date,pool=pool)
        FactorRisk, SpecialRisk = barra_engine.covariance_predict(start_date=date,frequency=params[2],Period=params[1])
        return {'factor_risk_short_m':FactorRisk.to_json(),'special_risk_short_m':SpecialRisk.to_json()}

    def get_FactorRisk_and_SpecialRisk_Long_W(self, param):
        config = ["Risk_Long_W"]
        params = config[0].split("_")
        pool = param['pool']
        date = parse(param['date']).strftime("%Y-%m-%d")
        barra_engine = MatrixModelCovariancePredictor(date=date,pool=pool)
        FactorRisk, SpecialRisk = barra_engine.covariance_predict(start_date=date,frequency=params[2],Period=params[1])
        return {'factor_risk_long_w':FactorRisk.to_json(),'special_risk_long_w':SpecialRisk.to_json()}

    def get_FactorRisk_and_SpecialRisk_Long_M(self, param):
        config = ["Risk_Long_M"]
        params = config[0].split("_")
        pool = param['pool']
        date = parse(param['date']).strftime("%Y-%m-%d")
        barra_engine = MatrixModelCovariancePredictor(date=date,pool=pool)
        FactorRisk, SpecialRisk = barra_engine.covariance_predict(start_date=date,frequency=params[2],Period=params[1])
        return {'factor_risk_long_m':FactorRisk.to_json(), 'special_risk_long_m':SpecialRisk.to_json()}

if __name__ == "__main__":
    # factor_return,special_return = Calculate().get_factor_return_and_special_return(date="20180223",pool="中证800")
    # print factor_return.head()
    # print special_return.head()
    risk = CalCulate().get_FactorRisk_and_SpecialRisk_Short_W(param = {"date":"20180319","pool":"中证800"})
    print(risk)