#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd


df = pd.read_csv("error_info.csv", encoding="utf-8")

df.to_excel("error_info.xlsx",  index=False)
