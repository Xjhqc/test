# -*- coding: UTF-8 -*-
import copy
from collections import OrderedDict
import time as time
import numpy as np
import pandas as pd
from dateutil.parser import parse
import json
from elasticsearch import Elasticsearch
import requests
import traceback
import _thread

from MatrixLibData import *
from ConstantPools import *


"""
功能：
    计算信号的归因分析、暴露分析
使用说明：
    1、将单个信号，或者多个时间序列上的回测结果信号信号格式(Code,EntryTime,ExitTime,Weight)，放在某一路径。
    2、选定选股股票池和基准

计算流程：
    1、获取因子在区间内的每日收益、个股在区间内的每日特质收益率
    2、将这些因子收益、特质收益累积起来
    3、计算区间期初的个股权重偏差
    4、将所有区间的收益贡献进行累积
"""


class MatrixModelAttribution():

    # --------------------------------------------------------------------
    def __init__(self, pool, benchmark,id="aaa",signal=None):
        self.data_engine = MatrixLibData('developer', '666666')  # 数据工具
        self.data_engine_fast = Elasticsearch(hosts=["http://192.168.134.191:9232"])
        self.trade_date = self.data_engine_fast.get(index="calendar", doc_type="calendar", id="cncalendar")['_source'][
            'items']
        self.pool = pool  # 因子暴露数据的股票池
        self.benchmark = benchmark  # 基准股票池
        self.id = id
        self.factor_list = RM_Factors
        self.sector_list = SW_Sectors
        self.input_signal = self.input_signal(signal) # signal #
        self.AttributeStartDate = self.input_signal["Date"].iloc[0]  # 记录组合起始交易日
        self.AttributeEndDate = self.input_signal["Date"].iloc[-1]  # 记录组合终止交易日
        self.get_FactorReturn_Fast()

    # --------------------------------------------------------------------
    def input_signal(self,signal):
        """
        获取组合信号(Code,Position,Date)
        输入希望是稠密的，空的Date则认为该交易日无持仓，否则先补全持仓数据
        """
        if(signal==None):
            signal = pd.read_csv("AttributeDemo(1).csv")
        signal["Position"] = signal["Position"] * 100
        return signal

    # --------------------------------------------------------------------
    def get_FactorReturn_Fast(self):
        s = time.time()
        self.factors_return = OrderedDict()
        self.special_return = OrderedDict()
        StartDate = parse(self.AttributeStartDate).strftime("%Y-%m-%d")
        EndDate = parse(self.AttributeEndDate).strftime("%Y-%m-%d")
        StartIndex = self.trade_date.index(StartDate) - 249
        EndIndex = self.trade_date.index(EndDate)
        TD250 = self.trade_date[StartIndex:(EndIndex + 1)]
        for day in TD250:
            d = parse(day).strftime("%Y%m%d")
            factor_return = self.data_engine_fast.get(index="cnfactor", doc_type="cnfactor",
                                                      id="csiall-" + d + "-factors_return")['_source']['items']
            special_return = self.data_engine_fast.get(index="cnfactor", doc_type="cnfactor",
                                                       id="csiall-" + d + "-special_return")['_source']['items']
            self.factors_return[day] = {i["key"]:i["value"] for i in factor_return}
            self.special_return[day] = {i["key"]: i["value"] for i in special_return}
        self.factors_return = pd.DataFrame(self.factors_return)
        self.special_return = pd.DataFrame(self.special_return)
        e = time.time()
        print("因子收益数据获取完毕用时%s" % (e - s))

    # --------------------------------------------------------------------
    def get_SupwinData_Fast(self, date):
        """计算单个截面组合信号与基准的个股权重差异"""
        # 根据组合的截面市值计算组合权重
        D = parse(date).strftime("%Y%m%d")  # 转换时间格式
        D2 = parse(date).strftime("%Y-%m-%d")  # 转换时间格式2
        Today_Position = self.input_signal.loc[self.input_signal["Date"] == date, :]  # 当日持仓

        # 读取数据
        Position = pd.DataFrame(
            self.data_engine_fast.get(index="cnfactor", doc_type="cnfactor", id="cnaall-" + D + "-close")['_source'][
                'items'])
        self.Bench_Return = \
            self.data_engine.get_quote(start_date=D, end_date=D, code=[symbols_map[self.benchmark]])["Returns"][0]
        benchmark_weight = pd.DataFrame(self.data_engine_fast.get(index="cnfactor", doc_type="cnfactor",
                                                                  id=coremap[self.benchmark] + "-" + D + "-weight")[
                                            '_source']['items'])
        rm_body = [coremap[self.pool] + "-" + D + "-" + fac for fac in RM_Fast_Factors]
        body = {
            "query": {
                "ids": {
                    "type": "cnfactor",
                    "values": rm_body
                }
            }
        }
        data = \
        self.data_engine_fast.search(index="cnfactor", doc_type="cnfactor", body=body, params={"size": 20})['hits'][
            'hits']
        res = OrderedDict()
        for dat in data:
            temp = {}
            var = dat['_id'].split("-")[-1]
            for val in dat['_source']['items']:
                if var != "supwin_sector":
                    temp[val['key']] = val["value"]
                else:
                    temp[val['key']] = val['stringValue']
            res[var] = temp
        expose_data = pd.DataFrame(res)
        expose_data.reset_index(inplace=True)
        factors_return = pd.DataFrame(self.data_engine_fast.get(index="cnfactor", doc_type="cnfactor",
                                                                id=coremap[self.pool] + "-" + D + "-factors_return")[
                                          '_source']['items'])
        special_return = pd.DataFrame(self.data_engine_fast.get(index="cnfactor", doc_type="cnfactor",
                                                                id=coremap[self.pool] + "-" + D + "-special_return")[
                                          '_source']['items'])
        TD250 = self.trade_date[self.trade_date.index(D2) - 249:self.trade_date.index(D2) + 1]

        # 获取当日持仓市值
        Position.columns = ["Code", "Close"]
        Position = pd.merge(Today_Position, Position, on="Code", how="inner")
        Position["Close"] = Position["Close"].values * Position["Position"].values
        Position["WeightPortfolio"] = Position["Close"] / Position["Close"].sum()
        self.Cap = Position["Close"].sum()
        benchmark_weight.columns = ["Code", "Weight"]
        benchmark_weight['Weight'] = benchmark_weight['Weight'] * 0.01
        merge_weight = pd.merge(benchmark_weight, Position, on='Code', how='outer').fillna(0)
        merge_weight['diff_weight'] = merge_weight.WeightPortfolio - merge_weight.Weight
        self.DiffWeight = merge_weight[["Code", "diff_weight"]]

        """计算单个截面主动风格暴露，通过组合与基准的个股权重配置差异"""
        # 1)准备当日风险因子数据
        expose_data = pd.get_dummies(expose_data, columns=['supwin_sector'])
        # 将个股权重差，与因子暴露数据merge
        expose_data = pd.merge(self.DiffWeight, expose_data, left_on='Code', right_on="index", how='inner')

        # 2)计算主动风格暴露
        expose_list = copy.deepcopy(RM_Fast_Factors)
        expose_list.extend([item.lower() for item in SW_Sectors])
        cols = [i for i in expose_data.columns if i in expose_list]
        active_expose = (np.matrix(expose_data.diff_weight) * np.matrix(expose_data[cols])).A1

        """对单个截面进行收益归因"""
        AE = pd.Series(active_expose, index=cols)
        # 获取因子收益
        factors_return = pd.Series(factors_return["value"].values,
                                   index=[it.lower() for it in factors_return["key"].values])
        special_return = pd.merge(self.DiffWeight, special_return, left_on="Code", right_on="key", how="inner")
        factors_return_attribute = factors_return * AE
        factors_return_attribute.pop("country")
        special_return_attribute = np.sum(special_return["diff_weight"] * special_return["value"])

        """对单个截面进行风险归因"""
        DiffWeight = pd.Series(self.DiffWeight["diff_weight"].values, index=self.DiffWeight["Code"].values)
        factors_return = self.factors_return.loc[:,TD250[0]:TD250[-1]].T
        factors_return.pop("Country")  # 去掉country列
        special_return = self.special_return.loc[:,TD250[0]:TD250[-1]].T
        special_return.dropna(how="all",inplace=True)
        factors_cov = factors_return.cov()
        factors_cov.index = [it.lower() for it in factors_cov.index]
        factors_cov.columns = [it.lower() for it in factors_cov.columns]
        special_var = np.var(special_return, axis=0)
        active_factor_risk = factors_cov.dot(AE) * AE
        active_special_risk = ((DiffWeight) ** 2 * special_var).fillna(0).sum()
        return pd.Series(active_expose, index=cols), factors_return_attribute, special_return_attribute, \
               np.sqrt(active_factor_risk * 250).fillna(0), np.sqrt(active_special_risk * 250)  # 返回年化风险，标准差为负意味着风险为0

    # --------------------------------------------------------------------
    def get_SupwinAttribute(self):
        """业绩归因"""
        active_expose = OrderedDict()
        factor_return_attribute = OrderedDict()
        special_return_attribute = OrderedDict()
        factor_risk_attribute = OrderedDict()
        special_risk_attribute = OrderedDict()
        self.total_Cap = {}
        self.benchmark_return = {}
        Date = np.unique(self.input_signal["Date"])
        for index, date in enumerate(Date):
            if (index+1)%((len(Date)+10-1)//10) == 0:
                print((index+1)/len(Date))
                # requests.get(url=Main.hostUrl+"/api/PerformanceAnalysis/SaveCalculationProgress?id="+str(self.id)+"&progress="+str((index+1)/len(Date)))
            AE, factors_return, special_return, factors_risk, special_risk = self.get_SupwinData_Fast(date=date)
            active_expose[date] = AE
            factor_return_attribute[date] = factors_return
            special_return_attribute[date] = special_return
            factor_risk_attribute[date] = factors_risk
            special_risk_attribute[date] = special_risk
            self.total_Cap[date] = self.Cap
            self.benchmark_return[date] = self.Bench_Return
        self.total_Cap = pd.Series(self.total_Cap)
        self.benchmark_return = pd.Series(self.benchmark_return)
        return active_expose, factor_return_attribute, special_return_attribute, \
               factor_risk_attribute, special_risk_attribute

    # --------------------------------------------------------------------
    def output1(self):
        output = {}
        AE, FRE, SRE, FRI, SRI = self.get_SupwinAttribute()
        Fri = FRI[self.AttributeEndDate]  # 最新一期的因子风险归因
        Sri = SRI[self.AttributeEndDate]  # 最新一期的特殊风险归因

        # 风险树状图
        RM_Fast_Factor = RM_Fast_Factors[:-1]
        FactorRiskContribution = Fri[RM_Fast_Factor].sum()
        SW_Fast_Sector = [i for i in Fri.index if i not in RM_Fast_Factors]
        SectorRiskContribution = Fri[SW_Fast_Sector].sum()
        SpecialRiskContribution = Sri
        ActiveRiskContribution = FactorRiskContribution + SectorRiskContribution + SpecialRiskContribution
        ActiveRiskPercent = 1
        FactorRiskPercent = FactorRiskContribution / ActiveRiskContribution
        SectorRiskPercent = SectorRiskContribution / ActiveRiskContribution
        SpecialRiskPercent = SpecialRiskContribution / ActiveRiskContribution
        output["Id"]=self.id
        output["IsSucess"]=True
        output["FactorRiskContribution"] = FactorRiskContribution
        output["SectorRiskContribution"] = SectorRiskContribution
        output["SpecialRiskContribution"] = SpecialRiskContribution
        output["ActiveRiskContribution"] = ActiveRiskContribution
        output["ActiveRiskPercent"] = ActiveRiskPercent
        output["FactorRiskPercent"] = FactorRiskPercent
        output["SectorRiskPercent"] = SectorRiskPercent
        output["SpecialRiskPercent"] = SpecialRiskPercent
        output["Benchmark"]=self.benchmark

        # 收益树状图
        Cap_return = (self.total_Cap / self.total_Cap.shift() - 1).fillna(0)
        TotalAnnualReturn = (Cap_return + 1).cumprod().iloc[-1] ** (250.0 / len(self.total_Cap)) - 1
        TotalReturnVolatility = Cap_return.std()
        benchmark_return = self.benchmark_return
        BenchMarkAnnualReturn = (benchmark_return + 1).cumprod().iloc[-1] ** (250.0 / len(benchmark_return)) - 1
        BenchMarkReturnVolatility = benchmark_return.std()
        ActiveAnnualReturn = TotalAnnualReturn - BenchMarkAnnualReturn
        ActiveReturnVolatility = (Cap_return - benchmark_return).std()
        FRE_df = pd.DataFrame(FRE)
        FactorAnnualReturnSep = (FRE_df.loc[RM_Fast_Factor] + 1).cumprod(axis=1).iloc[:, -1] ** (
                250.0 / len(self.total_Cap)) - 1
        SectorAnnualReturnSep = (FRE_df.loc[SW_Fast_Sector] + 1).cumprod(axis=1).iloc[:, -1] ** (
                250.0 / len(self.total_Cap)) - 1
        FactorReturnSum = FRE_df.loc[RM_Fast_Factor].sum()
        SectorReturnSum = FRE_df.loc[SW_Fast_Sector].sum()
        FactorAnnualReturn = FactorAnnualReturnSep.sum()
        FactorReturnVolatility = FactorReturnSum.std()
        SectorAnnualReturn = SectorAnnualReturnSep.sum()
        SectorReturnVolatility = SectorReturnSum.std()
        SpecialReturnAnnualReturn = ActiveAnnualReturn - FactorAnnualReturn - SectorAnnualReturn
        SpecialReturnReturnVolatility = (Cap_return - benchmark_return - FactorReturnSum - SectorReturnSum).std()
        output["TotalAnnualReturn"] = TotalAnnualReturn
        output["TotalReturnVolatility"] = TotalReturnVolatility
        output["BenchMarkAnnualReturn"] = BenchMarkAnnualReturn
        output["BenchMarkReturnVolatility"] = BenchMarkReturnVolatility
        output["ActiveAnnualReturn"] = ActiveAnnualReturn
        output["ActiveReturnVolatility"] = ActiveReturnVolatility
        output["FactorAnnualReturn"] = FactorAnnualReturn
        output["FactorReturnVolatility"] = FactorReturnVolatility
        output["SectorAnnualReturn"] = SectorAnnualReturn
        output["SectorReturnVolatility"] = SectorReturnVolatility
        output["SpecialReturnAnnualReturn"] = SpecialReturnAnnualReturn
        output["SpecialReturnReturnVolatility"] = SpecialReturnReturnVolatility

        # 所有持仓日的主动权重
        WeightCoefficient = []
        for date in AE.keys():
            dict_temp = {}
            dict_temp["Date"] = date
            Factor_list = []
            Sector_list = []
            for fac in RM_Fast_Factor:
                Factor_list.append({"FactorName": fac, "Value": AE[date][fac]})
            for sec in SW_Fast_Sector:
                Sector_list.append({"FactorName": sec.split("_")[-1], "Value": AE[date][sec]})
            dict_temp["Factors"] = Factor_list
            dict_temp["Sectors"] = Sector_list
            WeightCoefficient.append(dict_temp)
        output["WeightCoefficient"] = WeightCoefficient

        # 所有持仓日的主动收益
        ReturnCoefficient = []
        for date in FRE.keys():
            dict_temp = {}
            dict_temp["Date"] = date
            Factor_list = []
            Sector_list = []
            for fac in RM_Fast_Factor:
                Factor_list.append({"FactorName": fac, "Value": FRE[date][fac]})
            Factor_list.append({"FactorName": u"特质", "Value": SRE[date]})
            for sec in SW_Fast_Sector:
                Sector_list.append({"FactorName": sec.split("_")[-1], "Value": FRE[date][sec]})
            dict_temp["Factors"] = Factor_list
            dict_temp["Sectors"] = Sector_list
            ReturnCoefficient.append(dict_temp)
        output["ReturnCoefficient"] = ReturnCoefficient

        # 所有持仓日的主动风险
        RiskCoefficient = []
        for date in FRI.keys():
            dict_temp = {}
            dict_temp["Date"] = date
            Factor_list = []
            Sector_list = []
            for fac in RM_Fast_Factor:
                Factor_list.append({"FactorName": fac, "Value": FRI[date][fac]})
            Factor_list.append({"FactorName": u"特质", "Value": SRI[date]})
            for sec in SW_Fast_Sector:
                Sector_list.append({"FactorName": sec.split("_")[-1], "Value": FRI[date][sec]})
            dict_temp["Factors"] = Factor_list
            dict_temp["Sectors"] = Sector_list
            RiskCoefficient.append(dict_temp)
        output["RiskCoefficient"] = RiskCoefficient

        # 总的年化主动收益
        SumReturnCoefficient = {}
        SumReturnCoefficient["Date"] = self.input_signal["Date"].iloc[-1]
        Factor_list = []
        Sector_list = []
        for fac in RM_Fast_Factor:
            Factor_list.append({"FactorName": fac, "Value": FactorAnnualReturnSep[fac]})
        for sec in SW_Fast_Sector:
            Sector_list.append({"FactorName": sec.split("_")[-1], "Value": SectorAnnualReturnSep[sec]})
        Factor_list.append({"FactorName": u"特质", "Value": SpecialReturnAnnualReturn})  # 增加特质收益
        SumReturnCoefficient["Factors"] = Factor_list
        SumReturnCoefficient["Sectors"] = Sector_list
        output["SumReturnCoefficient"] = SumReturnCoefficient
        return json.dumps(output, indent=2)


def main():
    MatrixModelAttribution(pool="中证全指", benchmark="中证500").output1()
    # fl = open(u'./归因分析_建宏/Attribute.js', 'w')
    # fl.write("output=")
    # output = MatrixModelAttribution(pool="中证全指", benchmark="中证全指").output1()
    # fl.write(output)
    # fl.close()


# --------------------------------------------------------------------
if __name__ == '__main__':
    main()




#主类入口,提供外部调用#
class Main():
    hostUrl="http://localhost:30951"
    # --------------------------------------------------------------------
    def execute(self,params):
        _thread.start_new_thread(Main.task_execute, (self,params))
        return ""

    def task_execute(self,params):
        input_data = json.loads(params)
        benchmark=input_data["benchmark"]
        aid=input_data["id"]
        signal=pd.read_json(input_data["dataJson"])
        signal["Date"]=[d.replace("\"","") for d in signal["Date"]]
        
        output = {}
        output["Id"]=aid
        output["IsSucess"]=False
        print(str(aid)+"开始风险分析")
        
        try:
            output = MatrixModelAttribution(pool="中证全指",benchmark=benchmark,id=aid,signal=signal).output1()
        except Exception as e:
            msg = "error:"+ traceback.format_exc() 
            output["Info"]=msg
            output=json.dumps(output)
            print(msg)

        print(str(aid)+"结束风险分析")
        headers={'Content-Type':'application/json'} 
        cont=requests.post(Main.hostUrl+"/api/PerformanceAnalysis/SaveRiskFactor",headers=headers, data=output)
        requests.get(url=Main.hostUrl+"/api/PerformanceAnalysis/SaveCalculationProgress?id="+str(aid)+"&progress=1")
        print("发送结果成功")
        return output
