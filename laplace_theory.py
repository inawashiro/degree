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
        
    def s_theory(self, x_value):
        s1 = self.s1(x)
        s2 = self.s2(x)
        
        s_theory = np.ndarray((len(x_value),), 'object')
        s_theory[0] = s1
        s_theory[1] = s2
        
        for i in range(len(s_theory)):
            s_theory[i] = lambdify(x, s_theory[i], 'numpy')
            s_theory[i] = s_theory[i](x_value[0], x_value[1])
        
        return s_theory
    
    def a_theory(self, x_value):
        a = self.a(x)
        
        a_theory = np.ndarray((len(a),))
        for i in range(len(a)):
            temp = lambdify(x, a[i], 'numpy')
            a_theory[i] = temp(x_value[0], x_value[1])
        
        return a_theory
    
    def b_theory(self, x_value):
        b = self.b(x)
        
        b_theory = np.ndarray((len(b),))
        for i in range(len(b)):
            temp = lambdify(x, b[i], 'numpy')
            b_theory[i] = temp(x_value[0], x_value[1])
        
        return b_theory

    def r_theory(self, s_value):
        r = self.r(s)
        
        r_theory = np.ndarray((len(r),))
        for i in range(len(r)):
            temp = lambdify(s, r[i], 'numpy')
            r_theory[i] = temp(s_value[0], s_value[1])
        
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
    
    n = 5
    x_value = np.ndarray((2,))
    x_value_array = np.ndarray((n + 1, n + 1, 2,))
    s_theory_array = np.ndarray((n + 1, n + 1, 2,))
    a_theory_array = np.ndarray((n + 1, n + 1, 6,))
    b_theory_array = np.ndarray((n + 1, n + 1, 6,))
    r_theory_array = np.ndarray((n + 1, n + 1, 6,))
    for i in range(n + 1):
        for j in range(n + 1):
            x_value[0] = 1.0 + i/n
            x_value[1] = 1.0 + j/n
            
            s_theory = theory.s_theory(x_value)
            a_theory = theory.a_theory(x_value)
            b_theory = theory.b_theory(x_value)
            r_theory = theory.r_theory(s_theory)
            
            for k in range(len(x_value)):
                x_value_array[i][j][k] = x_value[k]
                s_theory_array[i][j][k] = s_theory[k]
            
            for k in range(len(a_theory)):
                a_theory_array[i][j][k] = a_theory[k]
                b_theory_array[i][j][k] = b_theory[k]
                r_theory_array[i][j][k] = r_theory[k]
                
    print('x_values = ')
    print(x_value_array)
    print('')
    
    print('s_theory = ')
    print(s_theory_array)
    print('')
    
    print('a_theory = ')
    print(a_theory_array)
    print('')
    
    print('b_theory = ')
    print(b_theory_array)
    print('')
    
    print('r_theory = ')
    print(r_theory_array)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    