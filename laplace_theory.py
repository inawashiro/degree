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

# For Constants
import math
from math import pi

# For Getting Access to Another Directory
import os

# For Measuring Computation Time
import time

# For Displaying Symbolic Notation
from IPython.display import display




class ProblemSettings():
    """ Define Principal Coordinate System """
    
    def s(self, x):
        """ Principal Coordinate System """
        """ s1 = Re{f(z)} & s2 = Im{f(z)} """
        
        s = np.ndarray((2,), 'object')
        """ f(z) = z**2 """
#        s[0] = x[0]**2 - x[1]**2
#        s[1] = 2*x[0]*x[1]
        """ f(z) = z**3 """
#        s[0] = x[0]**3 - 3*x[0]*x[1]**2
#        s[1] = -x[1]**3 + 3*x[0]**2*x[1]
        """ f(z) = z**4 """
#        s[0] = x[0]**4 - 6*x[0]**2*x[1]**2 + x[1]**4
#        s[1] = 4*x[0]**3*x[1] - 4*x[0]*x[1]**3
        """ f(z) = 1/(1 + z) """
        s[0] = (x[0] + 1)/((x[0] + 1)**2 + x[1]**2)
        s[1] = -x[1]/((x[0] + 1)**2 + x[1]**2)
        """ f(z) = exp(z) """
#        s[0] = exp(x[0])*sin(x[1])
#        s[1] = exp(x[0])*cos(x[1])
        """ f(z) = exp((π/2)z) """
#        s[0] = exp(pi/2*x[0])*sin(pi/2*x[1])
#        s[1] = exp(pi/2*x[0])*cos(pi/2*x[1])
        return s

    def u(self, s):
        """ Target Function under Principal Coordinate System """
        u = s[0]
        
        return u
    

class Verification(ProblemSettings):
    """ Verify Problem Settings """
    
    def __init__(self, x, s):
        self.ProblemSettings = ProblemSettings()
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
        
        ds_dx1 = np.ndarray((2,), 'object')
        ds_dx1[0] = diff(s[0], x[0])
        ds_dx1[1] = diff(s[1], x[0])
        ds_dx2 = np.ndarray((2,), 'object')
        ds_dx2[0] = diff(s[0], x[1])
        ds_dx2[1] = diff(s[1], x[1])
        
        g12 = dot(ds_dx1, ds_dx2)
        g12 = simplify(g12)
    
        return g12


class Plot(ProblemSettings):
    """ Display Plot """
    
    def __init__(self, x, s, x_value):
        self.ProblemSettings = ProblemSettings()
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
        x_value = self.x_value
        s_value = self.s_value()
        u_value = self.ProblemSettings.u(s_value)
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_wireframe(x_value[0], x_value[1], u_value, linewidth = 0.2)

        plt.savefig('target_function_3d.pdf')
        plt.savefig('target_function_3d.png')
        plt.pause(.01)
       
        return s_value
        
    def principal_coordinate_system_plot(self):
        x_value = self.x_value
        s_value = self.s_value()
            
        plt.gca().set_aspect('equal', adjustable='box')
        
        interval1 = np.arange(-100, 100, 1.0)
        interval2 = np.arange(-100, 100, 1.0)
        
        cr_s1 = plt.contour(x_value[0], x_value[1], s_value[0], interval1, colors = 'red')
#        levels1 = cr_s1.levels
#        cr_s1.clabel(levels1[::5], fmt = '%3.1f')
        
        cr_s2 = plt.contour(x_value[0], x_value[1], s_value[1], interval2, colors = 'blue')
#        levels2 = cr_s2.levels
#        cr_s2.clabel(levels2[::5], fmt = '%3.1f')
        
        plt.savefig('principal_coordinate_system.pdf')
        plt.savefig('principal_coordinate_system.png')
        plt.pause(.01)
        
    
class TheoreticalValue(ProblemSettings):
    """  Theoretical Values of Taylor Series Coefficients """
    
    def __init__(self, x, s, x_value):
        self.ProblemSettings = ProblemSettings()
        self.x = x
        self.s = s
        self.x_value = x_value
    
    def s_theory(self):
        """  Theoretical s_coordinate """
        x_value = self.x_value
        s_theory = self.ProblemSettings.s(x_value)
        
        return s_theory
    
    def a_theory(self):
        """  x_Taylor Series Coefficients of s1 """
        x = self.x
        s = self.ProblemSettings.s(x)
        x_value = self.x_value
        
        a_theory = np.ndarray((len(s), 6,), 'object')
        for i in range(len(s)):
            a_theory[i][0] = s[i]
            a_theory[i][1] = diff(s[i], x[0])
            a_theory[i][2] = diff(s[i], x[1])
            a_theory[i][3] = diff(s[i], x[0], 2)
            a_theory[i][4] = diff(s[i], x[0], x[1])
            a_theory[i][5] = diff(s[i], x[1], 2)
        
        for i in range(len(s)):
            for j in range(6):
                a_theory[i][j] = lambdify(x, a_theory[i][j], 'numpy')
                a_theory[i][j] = a_theory[i][j](x_value[0], x_value[1])
                
        return a_theory

    def r_theory(self):
        """ s_Taylor Series Coefficients of u """
        s = self.s
        u = self.ProblemSettings.u(s)
        s_theory = self.s_theory()
        
        r = np.ndarray((6,), 'object')
        r[0] = u
        r[1] = diff(u, s[0])
        r[2] = diff(u, s[1])
        r[3] = diff(u, s[0], 2)
        r[4] = diff(u, s[0], s[1])
        r[5] = diff(u, s[1], 2)
        
        r_theory = np.ndarray((len(r),), 'object')
        for i in range(len(r)):
            r_theory[i] = lambdify(s, r[i], 'numpy')
            r_theory[i] = r_theory[i](s_theory[0], s_theory[1])
        
        return r_theory   
        
    

        


if __name__ == '__main__':
    
    t0 = time.time()
    
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    ##################################
    Verification = Verification(x, s)
    ##################################
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
    
    
    x_value = np.meshgrid(np.arange(0, 2, 0.01), 
                          np.arange(0, 2, 0.01))
    
    ###########################
    Plot = Plot(x, s, x_value)
    ###########################
    os.chdir('./graph')
    
    print('3D Plot of u')
    Plot.u_plot()
    print('')
    
    print('Principal Coordinate System')
    Plot.principal_coordinate_system_plot()
    print('')    
    
    
#    n = 5
#    
#    x_value = np.ndarray((len(x),))
#    s_value = np.ndarray((len(s),))
#    
#    x_value_array = np.ndarray((n + 1, n + 1, len(x),))
#    s_theory_array = np.ndarray((n + 1, n + 1, len(s),))
#    a_theory_array = np.ndarray((n + 1, n + 1, len(s), 6))
#    r_theory_array = np.ndarray((n + 1, n + 1, 6,))
#    
#    ###################################################
#    TheoreticalValue = TheoreticalValue(x, s, x_value)
#    ###################################################  
#    
#    for i in range(n + 1):
#        for j in range(n + 1):
#            x_value[0] = 1.0 + i/n
#            x_value[1] = 1.0 + j/n
#        
#            s_theory = TheoreticalValue.s_theory()
#            a_theory = TheoreticalValue.a_theory()
#            r_theory = TheoreticalValue.r_theory()
#            
#            for k in range(len(x)):
#                x_value_array[i][j][k] = x_value[k]
#                
#            for k in range(len(s)):
#                s_theory_array[i][j][k] = s_theory[k]
#                
#                for l in range(6):
#                    a_theory_array[i][j][k][l] = a_theory[k][l]
#                
#            for k in range(6):
#                r_theory_array[i][j][k] = r_theory[k]
#                
#    
#                
#    print('x_values = ')
#    print(x_value_array)
#    print('')
#    
#    print('s_theory = ')
#    print(s_theory_array)
#    print('')
#    
#    print('a_theory = ')
#    print(a_theory_array)
#    print('')
#    
#    print('r_theory = ')
#    print(r_theory_array)
#    print('')
    
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                                                                                    