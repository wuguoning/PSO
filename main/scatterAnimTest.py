import matplotlib.pyplot as plt
import sys, os
import numpy as np
import pandas as pd
sys.path.append('../')

from src.scatterAnim import AnimatedScatter
def sphereFunc(x,y):
    return x**2/20. + y**2

bounds = [-10, 10, -5, 5]
#=========================================
# load data
df = pd.read_csv('../data/sphereSwarmHistory.csv')
dlen = len(df) # data length
xydata=[]
for i in range(dlen):
    dtmp = df.loc[i]
    dtmp1 = [[float(x.strip("[]")) for x in dtmp[j].split()] for j in range(len(dtmp))]
    xydata.append(np.array(dtmp1).T)

numpoint = np.shape(xydata)[-1]
#print(numpoint)

# animation
ani = AnimatedScatter(numpoint, xydata, dlen, sphereFunc, bounds)
plt.show()
