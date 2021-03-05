# -*- coding: utf-8 -*-

#=================================================#
"""
Simple Particle Swarm Optimization (PSO) with Python
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   2021-01-03

Reference:
    1. Based on: Nathan A. Rooy. Simple Particle Swarm Optimization (PSO) with Python
    2. lx青萍之末,CSND, 最优化算法之粒子群算法（PSO)
    3. J. Kennedy, R Eberhart. Particle Swarm Optimization. Proceedings of the IEEE International Conference on Neural Networks, 1995:1942-1948
"""

#=================================================#
#IMPORT DEPENDENCIES

from __future__ import division
import random
import math
import os,sys
sys.path.append('../')

from src.partSwarmOptim import Particle, SwarmOptim

#=================================================#
# function we are attempting to optimize (minimize)
def sphereFunc (x):
    total = 0
    a = [20., 1.]
    for i in range(len(x)):
        total += x[i]**2/a[i]
    return total

#=================================================#
if __name__ == "__main__":

    initial=[5.,5.]               # initial starting location [x1,x2...]
    bounds=[(-10.,10.),(-10.,10.)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    path = '../data'
    savefile = os.path.join(path, "sphereSwarmHistory.csv")

    obj = SwarmOptim(savefile)
    obj.simplePSO(sphereFunc,initial,bounds,num_particles=200,maxiter=30, verbose=1)
