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

# For Symbolic Notation
import sympy as sym
from sympy import Symbol, diff, nsolve, lambdify
sym.init_printing()

# For Symbolic Expression Displaying
from IPython.display import display

# For Getting Access to Another Directory
import os

# For measuring computation time
import time



class Theory():
    """" Analytical Experessions of Parameters """
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    def __init__(self):
        s_interval = np.ndarray((2,))
        s_interval[0] = 2.5
        s_interval[1] = 2.5
        self.s_interval = s_interval
    
        number_of_points_on_line = np.ndarray((2,), 'int')
        number_of_points_on_line[0] = round(12.5/s_interval[0])
        number_of_points_on_line[1] = round(12.5/s_interval[1])
        self.number_of_points_on_line = number_of_points_on_line

    def s1(self, x):
        s1 = x[0]**3 - 3*x[0]*x[1]**2
        
        return s1

    def s2(self, x):
        s2 = -x[1]**3 + 3*x[0]**2*x[1]
        
        return s2

    def u(self, s):
        """ One of the Polynomial Solutions of Laplace Equation """
        u = s[0]
        
        return u

    def a(self, x):
        """  Theoretical Values of Taylor Series of s1 w.r.t. x """
        s1 = self.s1(x)
        a = np.ndarray((6,), 'object')
        a[0] = s1
        a[1] = diff(s1, x[0])
        a[2] = diff(s1, x[1])
        a[3] = diff(s1, x[0], 2)
        a[4] = diff(s1, x[0], x[1])
        a[5] = diff(s1, x[1], 2)
        
        return a
    
    def b(self, x):
        """  Theoretical Values of Taylor Series of s w.r.t. x """
        s2 = self.s2(x)
        b = np.ndarray((6,), 'object')
        b[0] = s2
        b[1] = diff(s2, x[0])
        b[2] = diff(s2, x[1])
        b[3] = diff(s2, x[0], 2)
        b[4] = diff(s2, x[0], x[1])
        b[5] = diff(s2, x[1], 2)
        
        return b
    
    def r(self, s):
        """ Theoretical Values of Taylor Series of u w.r.t. x """
        u = self.u(s)
        r = np.ndarray((6,), 'object')
        r[0] = u
        r[1] = diff(u, s[0])
        r[2] = diff(u, s[1])
        r[3] = diff(u, s[0], 2)
        r[4] = diff(u, s[0], s[1])
        r[5] = diff(u, s[1], 2)
        
        return r
    
    def s_values(self):
        s1_interval = self.s_interval[0]
        s2_interval = self.s_interval[1]
        number_of_points_on_line = self.number_of_points_on_line
        s_values = np.ndarray((number_of_points_on_line[1],
                               number_of_points_on_line[0], 
                               2))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                s_values[i][j][0] = s1_interval*(- j - 1)
                s_values[i][j][1] = s2_interval*(i + 1)
                
        return s_values

    def x_values(self, x):
        number_of_points_on_line = self.number_of_points_on_line
        s1 = self.s1(x)
        s2 = self.s2(x)
        x_values = np.ndarray((number_of_points_on_line[1],
                               number_of_points_on_line[0], 
                               2))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                s_values = self.s_values()[i][j]
                f1 = s1 - s_values[0]
                f2 = s2 - s_values[1]
                solution = nsolve((f1, f2), x, (1.5, 1.5))
                x_values[i][j][0] = solution[0]
                x_values[i][j][1] = solution[1]
                
        return x_values
    
    def a_theory(self, x):
        number_of_points_on_line = self.number_of_points_on_line
        a = self.a(x)
        a_theory = np.ndarray((number_of_points_on_line[1],
                               number_of_points_on_line[0], 
                               6))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                x_value = self.x_values(x)[i][j]
                for k in range(len(a)):
                    temp = lambdify(x, a[k], 'numpy')
                    a_theory[i][j][k] = temp(x_value[0], x_value[1])
        
        return a_theory
    
    def b_theory(self, x):
        number_of_points_on_line = self.number_of_points_on_line
        b = self.b(x)
        b_theory = np.ndarray((number_of_points_on_line[1],
                               number_of_points_on_line[0], 
                               6))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                x_value = self.x_values(x)[i][j]
                for k in range(len(b)):
                    temp = lambdify(x, b[k], 'numpy')
                    b_theory[i][j][k] = temp(x_value[0], x_value[1])
        
        return b_theory

    def r_theory(self, s):
        number_of_points_on_line = self.number_of_points_on_line
        r = self.r(s)
        r_theory = np.ndarray((number_of_points_on_line[1],
                               number_of_points_on_line[0], 
                               6))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                s_value = self.s_values()[i][j]
                for k in range(len(r)):
                    temp = lambdify(s, r[k], 'numpy')
                    r_theory[i][j][k] = temp(s_value[0], s_value[1]) 
                
        return r_theory    
        
    
class Plot(Theory):
    """ Display Plot """
    
    x = np.meshgrid(np.arange(1, 2, 0.01),
                    np.arange(1, 2, 0.01))
    
    def __init__(self):
        self.theory = Theory()
    
    def u_plot(self, x):
        u = self.theory.u(s)
        s1 = self.theory.s1(x)
        s2 = self.theory.s2(x)
        u = lambdify(s, u, 'numpy')
        u = u(s1, s2)
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_wireframe(x[0], x[1], u, linewidth = 0.2)

        plt.savefig('target_function_3d.pdf')
        plt.savefig('target_function_3d.png')
        plt.pause(.01)
        
    def principal_coordinate_system_plot(self, x):
        s1 = self.theory.s1(x)
        s2 = self.theory.s2(x)
            
        plt.gca().set_aspect('equal', adjustable='box')
        
        interval1 = np.arange(-15.0, 2.5, 2.5)
        interval2 = np.arange(0.0, 15.0, 2.5)
        
        cr_s1 = plt.contour(x[0], x[1], s1, interval1, colors = 'red')
        levels1 = cr_s1.levels
        cr_s1.clabel(levels1[::2], fmt = '%3.1f')
        
        cr_s2 = plt.contour(x[0], x[1], s2, interval2, colors = 'blue')
        levels2 = cr_s2.levels
        cr_s2.clabel(levels2[::2], fmt = '%3.1f')
        
        plt.savefig('principal_coordinate_system.pdf')
        plt.savefig('principal_coordinate_system.png')
        plt.pause(.01)
        


if __name__ == '__main__':
    
    t0 = time.time()
    
    theory = Theory()
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)

    print('x_values = ')
    print(theory.x_values(x))
    print('')
    
    print('a_theory = ')
    print(theory.a_theory(x))
    print('')
    
    print('b_theory = ')
    print(theory.b_theory(x))
    print('')
    
    print('r_theory = ')
    print(theory.r_theory(s))
    print('')
    
    plot = Plot()
    
    os.chdir('./graph')
    
    x = np.meshgrid(np.arange(1, 2, 0.01),
                    np.arange(1, 2, 0.01))
    
    print('3D Plot of u')
    plot.u_plot(x)
    print('')
    
    print('Principal Coordinate System')
    plot.principal_coordinate_system_plot(x)
    print('')    
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    