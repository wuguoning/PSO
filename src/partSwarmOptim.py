# -*- coding: utf-8 -*-

#=================================================#
"""
Simple Particle Swarm Optimization (PSO) with Python
Author: Gordon Woo
Email:  wuguoning@gmail.com
Date:   2021-01-03

Reference:
    1. Based on: Nathan A. Rooy. Simple Particle Swarm
       Optimization (PSO) with Python
    2. lx青萍之末,CSND, 最优化算法之粒子群算法（PSO)
    3. J. Kennedy, R Eberhart. Particle Swarm Optimization.
       Proceedings of the IEEE International Conference on
       Neural Networks, 1995:1942-1948
"""

#=================================================#
#IMPORT DEPENDENCIES
import numpy as np
import random
import math
import pandas as pd

np.set_printoptions(formatter={'float': '{:0.6f}'.format})

#=================================================#
# Particle initialize

class Particle(object):
    def __init__(self,x0):
        self.position_i = []          # particle position
        self.velocity_i = []          # particle velocity
        self.pos_best_i = []          # best position individual
        self.err_best_i = -1          # best error individual
        self.err_i = -1               # error individual
        self.x0 = x0

        for i in range(0, len(self.x0)):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])


    # evaluate current fitness
    def evaluate(self,costFunc):
        """
        costFunc: test function
        """
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        """
        Parameters:
          pos_best_g: group best position
        """
        w = 0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1        # cognative constant
        c2 = 2        # social constant

        for i in range(0, len(self.x0)):
            r1=random.random()
            r2=random.random()

            vel_cognitive = c1*r1*(self.pos_best_i[i] - self.position_i[i])
            vel_social = c2*r2*(pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w*self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        """
        Parameters:
          bounds: the value bounds
        """
        for i in range(0,len(self.x0)):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]



#=================================================#
class SwarmOptim(object):

    def __init__(self, savefile):
        self.savefile = savefile

    def simplePSO(self,costFunc,x0,bounds,num_particles,maxiter, verbose):
        """
        Simple Partical Swarm Optimization Method
        Parameters:
            consFunc: test function
            x0:       initial position
            bounds:   bounds of position
            num_particles: particle number
            maxiter:  maximum iteration number
            verbose:  if y,print, else nonprint
        """
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        data = []
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0+1.5*np.random.random(np.shape(x0))))
            data.append([])
            data[i].append(np.array(swarm[i].position_i))
            #print(data[i])

        # begin optimization loop
        i=0
        while i < maxiter:
            if verbose:
                print(f'iter: {i:>4d}, best solution: {err_best_g:10.6f}')

            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
                data[j].append(np.array(swarm[j].position_i))
            i+=1

        ##================================================
        ## A check point window

        #for i in range(0,num_particles):
        #    print(data[i])
        ##================================================

        # save to a pandas
        ind = []
        for i in range(num_particles):
            imf = 'swarm{}'.format(i)
            ind.append(imf)
        df = pd.DataFrame([list(i) for i in zip(*data)], columns=ind)
        df.to_csv(self.savefile, index=False)

        if verbose:
            ## print final results
            print("\n\n")
            print("="*50)
            print("The Final solution and position is:\n")
            print(str(pos_best_g))
            print(str(err_best_g))
            print("="*50)

