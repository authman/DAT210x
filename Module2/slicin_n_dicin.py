# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:49:33 2017

@author: pcurzytek
"""
import pandas as pd

df = pd.read_csv('Datasets/direct_marketing.csv')

print df.loc[0:4, ['history', 'mens']]