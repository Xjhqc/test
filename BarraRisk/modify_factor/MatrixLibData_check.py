#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @File  : MatrixLibData_check.py
# @Author: Rocket
# @Date  :
from lib.MatrixLibData_old import MatrixLibData as MatrixLibData_old
from lib.MatrixLibData import MatrixLibData as MatrixLibData_new
from lib.ConstantPools import *

import pandas as pd


class MatrixLibData_check(object):
    def __init__(self):
        self.DataTool_old = MatrixLibData_old(username="xiejie",password="matrix")
        self.DataTool_new = MatrixLibData_new(username="developer",password="666666")

        # alpha191没有以下代号的因子
        gtja_no_data = [9, 10, 30, 68, 111, 143, 144, 154, 162, 164, 166]
        # 国君因子列表
        self.gtja_factor_list = ['GTJA_Alpha' + str(i) for i in range(1, 192) if i not in gtja_no_data]
        self.WQ101_factor_list = ['alpha_' + str(i) for i in range(1, 102)]
        self.WQ101_factor_list.remove('alpha_7')
        self.WQ101_factor_list1 = ['Alpha#' + str(i) for i in range(1, 102)]
        self.WQ101_factor_list1.remove('Alpha#7')

        # self.factors = RAW_factors
        self.factors = self.WQ101_factor_list
        self.start = "20180330"
        self.end = "20180330"
        self.period = "M"

    # 批量检查
    def check_amount(self):
        date_month = self.DataTool_old.get_trade_date(start_date=self.start,end_date=self.end,period=self.period)
        result = {}
        for index,fac in enumerate(self.factors):
            fac_dif = []
            for d in date_month:
                # print(self.WQ101_factor_list1[index],fac)
                data_old = self.DataTool_old.get_factor(date=d,config=[self.WQ101_factor_list1[index]],pool="中证800")
                # data_old = self.DataTool_new.get_factor(date=d, config=[fac], pool="中证800")
                data_new = self.DataTool_new.get_factor(date=d,config=[fac],pool="中证800")
                data = pd.merge(data_old,data_new,on="Code",how="outer")
                # dif = (data[self.gtja_factor_list[index]] - data[fac]).abs().sum()
                dif = (data[self.WQ101_factor_list1[index]]-data[fac]).abs().sum()
                # dif = (data[fac] - data[fac]).abs().sum()
                fac_dif.append(dif)
                if dif>0.01:
                    print(fac+" in "+d+" is different",dif)
            result[fac] = dif
            if sum(fac_dif)<0.0001:
                print(fac+" is OK!!!")
        print(pd.Series(result).sort_values())

    def check_single(self,factor,date,pool):
        factor_old ="Alpha#93"
        factor_new ="alpha_93"
        data_old = self.DataTool_old.get_factor(date=date, config=[factor_old], pool=pool)
        data_old[factor_old+"_old"] = data_old[factor_old]
        data_new = self.DataTool_new.get_factor(date=date, config=[factor_new], pool=pool)
        data_new[factor_new+"_new"] = data_new[factor_new]
        data = pd.merge(data_old[["Code",factor_old+"_old","ValueDate"]],data_new[["Code",factor_new+"_new"]],on="Code",how="outer")
        data["diff"] = data[factor_old+"_old"]-data[factor_new+"_new"]
        # print((data[factor + "_old"] - data[factor + "_new"]).abs().sum())
        # print(data.loc[data.Code=="300059.SZ",])
        print(data.sort_values(by="diff"))

def main():
    MatrixLibData_check().check_single(factor="GTJA_Alpha1",date="2018-03-30",pool="中证800")
    # MatrixLibData_check().check_amount()

if __name__ == '__main__':
    main()