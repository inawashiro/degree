# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:01:06 2018

@author: inawashiro
"""

# For Visualization
import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle']='solid'

# For 3D Graph
from mpl_toolkits.mplot3d import Axes3D

# For Numerical Computation 
import numpy as np
from numpy import dot

# For Symbolic Notation
import sympy as sym
from sympy import Symbol, diff, lambdify, simplify, exp, sin, cos
sym.init_printing()

# For Displaying Symbolic Notation
from IPython.display import display

# For Constants
import math
from math import pi

# For Getting Access to Another Directory
import os

# For Measuring Computation Time
import time




class ProblemSettings():
    """ Define Principal Coordinate System """
    
    def __init__(self, f_id):
        self.f_id = f_id
        
    def s(self, x):
        """ Principal Coordinate System """
        """ s1 = Re{f(z)} & s2 = Im{f(z)} """
        f_id = self.f_id
        
        s = np.ndarray((2,), 'object')
        
        if f_id == 'z^2':
            s[0] = x[0]**2 - x[1]**2
            s[1] = 2*x[0]*x[1]        
        if f_id == 'z^3':
            s[0] = x[0]**3 - 3*x[0]*x[1]**2
            s[1] = -x[1]**3 + 3*x[0]**2*x[1]        
        if f_id == 'z^4':
            s[0] = x[0]**4 - 6*x[0]**2*x[1]**2 + x[1]**4
            s[1] = 4*x[0]**3*x[1] - 4*x[0]*x[1]**3
        if f_id == 'exp(z)':
            s[0] = exp(x[0])*sin(x[1])
            s[1] = exp(x[0])*cos(x[1])
    
        return s

    def u(self, s):
        """ Target Function under Principal Coordinate System """
        u = s[0]
        
        return u
    

class Verification(ProblemSettings):
    """ Verify Problem Settings """
    
    def __init__(self, f_id, x, s):
        self.ProblemSettings = ProblemSettings(f_id)
        self.x = x
        self.s = s
        
    def laplacian_u(self):
        """ Verify Δu = 0 """
        x = self.x
        s = self.ProblemSettings.s(x)
        u = self.u(s)
        
        laplacian_u = diff(u, x[0], 2) + diff(u, x[1], 2)
        laplacian_u = simplify(laplacian_u)
    
        return laplacian_u
    
    def du_ds2(self):
        """ Verify Δu = 0 """
        s = self.s
        u = self.ProblemSettings.u(s)
        
        du_ds2 = diff(u, s[1])
    
        return du_ds2
    
    def g12(self):
        """ Verify g_12 = 0 """
        x = self.x
        s = self.ProblemSettings.s(x)
        
        ds1_dx = np.ndarray((2,), 'object')
        ds1_dx[0] = diff(s[0], x[0])
        ds1_dx[1] = diff(s[0], x[1])
        ds2_dx = np.ndarray((2,), 'object')
        ds2_dx[0] = diff(s[1], x[0])
        ds2_dx[1] = diff(s[1], x[1])
        
        g12 = dot(ds1_dx, ds2_dx)
        g12 = simplify(g12)
    
        return g12


class Plot(ProblemSettings):
    """ Display Plot """
    
    def __init__(self, f_id, x, s, x_value):
        self.ProblemSettings = ProblemSettings(f_id)
        self.f_id = f_id
        self.x = x
        self.s = s
        self.x_value = x_value
    
    def s_value(self):
        x = self.x
        s = self.ProblemSettings.s(x)
        x_value = self.x_value
        
        s_value = np.ndarray((len(s),), 'object')
        s_value = lambdify(x, s, 'numpy')
        s_value = s_value(x_value[0], x_value[1])
        
        return s_value
    
    def u_plot(self):
        f_id = self.f_id
        x_value = self.x_value
        s_value = self.s_value()
        u_value = self.ProblemSettings.u(s_value)
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_wireframe(x_value[0], x_value[1], u_value, linewidth = 0.2)
        
        plt.locator_params(axis='x',nbins=5)
        plt.locator_params(axis='y',nbins=5)
        plt.locator_params(axis='z',nbins=5)

        plt.savefig('../graph/' + f_id + '/3d_plot.pdf')
        plt.savefig('../graph/' + f_id + '/3d_plot.png')
        
        plt.pause(.01)
        
    def principal_coordinate_system_plot(self):
        f_id = self.f_id
        x_value = self.x_value
        s_value = self.s_value()
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        interval1 = np.arange(-100, 100, 1.0)
        interval2 = np.arange(-100, 100, 1.0)
        
        plt.contour(x_value[0], x_value[1], s_value[0], interval1, colors = 'red')        
        plt.contour(x_value[0], x_value[1], s_value[1], interval2, colors = 'blue')
        
        plt.locator_params(axis='x',nbins=5)
        plt.locator_params(axis='y',nbins=5)
        
        plt.savefig('../graph/' + f_id + '/principal_coordinate_system.pdf')
        plt.savefig('../graph/' + f_id + '/principal_coordinate_system.png')
        
        plt.pause(.01)
        
    
class TheoreticalValue(ProblemSettings):
    """  Theoretical Values of Taylor Series Coefficients """
    
    def __init__(self, f_id, x, s, x_value):
        self.ProblemSettings = ProblemSettings(f_id)
        self.f_id = f_id
        self.x = x
        self.s = s
        self.x_value = x_value
    
    def s_theory(self):
        """  Theoretical s_coordinate """
        x_value = self.x_value
        s_theory = self.ProblemSettings.s(x_value)
        
        return s_theory
    
    def s_coeff_theory(self):
        """  x_Taylor Series Coefficients of s1 """
        x = self.x
        s = self.ProblemSettings.s(x)
        x_value = self.x_value
        
        s_coeff_theory = np.ndarray((len(s), 6,), 'object')
        for i in range(len(s)):
            s_coeff_theory[i][0] = s[i]
            s_coeff_theory[i][1] = diff(s[i], x[0])
            s_coeff_theory[i][2] = diff(s[i], x[1])
            s_coeff_theory[i][3] = diff(s[i], x[0], 2)
            s_coeff_theory[i][4] = diff(s[i], x[0], x[1])
            s_coeff_theory[i][5] = diff(s[i], x[1], 2)
            
        for i in range(len(s)):
            for j in range(len(s_coeff_theory[i])):
                s_coeff_theory[i][j] = lambdify(x, s_coeff_theory[i][j], 'numpy')
                s_coeff_theory[i][j] = s_coeff_theory[i][j](x_value[0], x_value[1])
                
        return s_coeff_theory

    def u_coeff_theory(self):
        """ s_Taylor Series Coefficients of u """
        s = self.s
        u = self.ProblemSettings.u(s)
        s_theory = self.s_theory()
        
        u_coeff_theory = np.ndarray((6,), 'object')
        u_coeff_theory[0] = u
        u_coeff_theory[1] = diff(u, s[0])
        u_coeff_theory[2] = diff(u, s[1])
        u_coeff_theory[3] = diff(u, s[0], 2)
        u_coeff_theory[4] = diff(u, s[0], s[1])
        u_coeff_theory[5] = diff(u, s[1], 2)
        
        for i in range(len(u_coeff_theory)):
            u_coeff_theory[i] = lambdify(s, u_coeff_theory[i], 'numpy')
            u_coeff_theory[i] = u_coeff_theory[i](s_theory[0], s_theory[1])
        
        return u_coeff_theory   
        


if __name__ == '__main__':
    
    t0 = time.time()
    
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    ####################
#    f_id = 'z^2'
#    f_id = 'z^3'
#    f_id = 'z^4'
    f_id = 'exp(z)'
    ####################
    
    print('')
    print('f(z) = ', f_id)
    print('')
    
    #######################################
    Verification = Verification(f_id, x, s)
    #######################################
    laplacian_u = Verification.laplacian_u()
    du_ds2 = Verification.du_ds2()
    g12 = Verification.g12()
    
    print('Δu = ')
    display(laplacian_u)
    print('')
    
    print('du_ds2 = ')
    display(du_ds2)
    print('')
    
    print('g12 = ')
    display(g12)
    print('')
    
    x_min = np.ndarray((2))
    x_min[0] = 0.0
    x_min[1] = 0.0
    
    x_max = np.ndarray((2))
    x_max[0] = 2.0
    x_max[1] = 2.0
    
    x_sidelength = np.ndarray((2))
    x_sidelength[0] = x_max[0] - x_min[0]
    x_sidelength[1] = x_max[1] - x_min[1]
    
    x_value = np.meshgrid(np.arange(x_min[0], x_max[0], (x_sidelength[0])/500), 
                          np.arange(x_min[1], x_max[1], (x_sidelength[1])/500))
    
    #################################
    Plot = Plot(f_id, x, s, x_value)
    #################################
    os.chdir('./graph')
    
    print('3D Plot of u')
    Plot.u_plot()
    print('')
    
    print('Principal Coordinate System')
    Plot.principal_coordinate_system_plot()
    print('')    
    
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
    
    
    
    
    
    
    
    
    
    
    