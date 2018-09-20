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

# For Symbolic Expression Display
from IPython.display import display

# For Getting Access to Another Directory
import os

# For measuring computation time
import time



class Theory():
    """" Analytical Experessions of Parameters """
    x = [Symbol('x_1', real = True), 
         Symbol('x_2', real = True)
         ]
    
    def __init__(self):
        self.s1_interval = 12.5
        self.s2_interval = 12.5
        self.number_of_points_on_line = [round(12.5/self.s1_interval), 
                                         round(12.5/self.s2_interval)]
       
    def u(self, x):
        """ One of the Polynomial Solutions of Laplace Equation """
        return x[0]**3 - 3*x[0]*x[1]**2

    def s1(self, x):
        u = self.u(x)
        return u

    def s2(self, x):
        return -x[1]**3 + 3*x[0]**2*x[1]

    def r(self):
        """ Theoretical Values of Taylor Series of u w.r.t. x """
        return [0, 1, 0, 0, 0, 0]

    def a(self, x):
        """  Theoretical Values of Taylor Series of s1 w.r.t. x """
        s1 = self.s1(x)
        return [s1, 
                diff(s1, x[0]), 
                diff(s1, x[1]),
                diff(s1, x[0], 2)/2, 
                diff(s1, x[0], x[1]), 
                diff(s1, x[1], 2)/2
                ]
    
    def b(self, x):
        """  Theoretical Values of Taylor Series of s w.r.t. x """
        s2 = self.s2(x)
        return [s2, 
                diff(s2, x[0]), 
                diff(s2, x[1]),
                diff(s2, x[0], 2)/2, 
                diff(s2, x[0], x[1]), 
                diff(s2, x[1], 2)/2
                ]
    
    def s_values(self):
        s1_interval = self.s1_interval
        s2_interval = self.s2_interval
        number_of_points_on_line = self.number_of_points_on_line
        temp = np.ndarray((number_of_points_on_line[1],number_of_points_on_line[0], 2))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                temp[i][j][0] = s1_interval*(- j - 1)
                temp[i][j][1] = s2_interval*(i + 1)
        return temp

    def x_values(self, x):
        number_of_points_on_line = self.number_of_points_on_line
        s1 = self.s1(x)
        s2 = self.s2(x)
        temp = np.ndarray((number_of_points_on_line[1],number_of_points_on_line[0], 2))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                s_values = self.s_values()[i][j]
                f1 = s1 - s_values[0]
                f2 = s2 - s_values[1]
                solution = nsolve((f1, f2), x, (1.5, 1.5))
                temp[i][j][0] = solution[0]
                temp[i][j][1] = solution[1]
        return temp
    
    def r_theory(self, x):
        number_of_points_on_line = self.number_of_points_on_line
        r = self.r()
        temp = np.ndarray((number_of_points_on_line[1],number_of_points_on_line[0], 6))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                temp[i][j][0] = r[0]
                temp[i][j][1] = r[1]
                temp[i][j][2] = r[2]
                temp[i][j][3] = r[3]
                temp[i][j][4] = r[4]
                temp[i][j][5] = r[5]
        return temp
    
    def a_theory(self, x):
        number_of_points_on_line = self.number_of_points_on_line
        a = self.a(x)
        temp = np.ndarray((number_of_points_on_line[1],number_of_points_on_line[0], 6))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                x_value = self.x_values(x)[i][j]
                a0 = lambdify(x, a[0], 'numpy')
                a1 = lambdify(x, a[1], 'numpy')
                a2 = lambdify(x, a[2], 'numpy')
                a3 = lambdify(x, a[3], 'numpy')
                a4 = lambdify(x, a[4], 'numpy')
                a5 = lambdify(x, a[5], 'numpy')
                temp[i][j][0] = a0(x_value[0], x_value[1])
                temp[i][j][1] = a1(x_value[0], x_value[1])
                temp[i][j][2] = a2(x_value[0], x_value[1])
                temp[i][j][3] = a3(x_value[0], x_value[1])
                temp[i][j][4] = a4(x_value[0], x_value[1])
                temp[i][j][5] = a5(x_value[0], x_value[1])
        return temp
    
    def b_theory(self, x):
        number_of_points_on_line = self.number_of_points_on_line
#        number_of_points_on_line = Theory.number_of_points_on_line
        b = self.b(x)
        temp = np.ndarray((number_of_points_on_line[1],number_of_points_on_line[0], 6))
        for i in range(number_of_points_on_line[0]):
            for j in range(number_of_points_on_line[1]):
                x_value = self.x_values(x)[i][j]
                b0 = lambdify(x, b[0], 'numpy')
                b1 = lambdify(x, b[1], 'numpy')
                b2 = lambdify(x, b[2], 'numpy')
                b3 = lambdify(x, b[3], 'numpy')
                b4 = lambdify(x, b[4], 'numpy')
                b5 = lambdify(x, b[5], 'numpy')
                temp[i][j][0] = b0(x_value[0], x_value[1])
                temp[i][j][1] = b1(x_value[0], x_value[1])
                temp[i][j][2] = b2(x_value[0], x_value[1])
                temp[i][j][3] = b3(x_value[0], x_value[1])
                temp[i][j][4] = b4(x_value[0], x_value[1])
                temp[i][j][5] = b5(x_value[0], x_value[1])
        return temp

        
        
class Plot(Theory):
    """ Display Plot """
    
    x = np.meshgrid(np.arange(1, 2, 0.01),
                    np.arange(1, 2, 0.01))
    
    def __init__(self):
        self.theory = Theory()
    
    def u_plot(self):
        u = self.theory.u(x)
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_wireframe(x[0], x[1], u, linewidth = 0.2)

        plt.savefig('target_function_3d.pdf')
        plt.savefig('target_function_3d.png')
        plt.pause(.01)
        
        
    def principal_coordinate_system_plot(self):
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
    
#    x = [Symbol('x_1', real = True), 
#         Symbol('x_2', real = True)
#         ]
    
    print('u = ')
    display(theory.u(x))
    print('')

    print('x_values = ', theory.x_values(x))
    print('')
    
    print('r_theory = ', theory.r_theory(x))
    print('')
    
    print('a_theory = ', theory.a_theory(x))
    print('')
    
    print('b_theory = ', theory.b_theory(x))
    print('')
    
    
    plot = Plot()
    
    os.chdir('./graph')
    
#    x = np.meshgrid(np.arange(1, 2, 0.01),
#                    np.arange(1, 2, 0.01))
    
    print('3D Plot of u')
    plot.u_plot()
    print('')
    
    print('Principal Coordinate System')
    plot.principal_coordinate_system_plot()
    print('')    
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
    
    
    
    
    
    
    
    
    
    
    
    