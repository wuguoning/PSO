import numpy as np
import pandas as pd
import sys,os
import csv

np.set_printoptions(formatter={'float': '{:0.6f}'.format})

sys.path.append('../')

df = pd.read_csv('../data/swarmHistory.csv')
xydata = df.loc[0]
print(xydata)
d1 = [[float(x.strip("[]")) for x in xydata[i].split()] for i in range(len(xydata))]
print(np.array(d1).T)
