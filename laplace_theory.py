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
from sympy import Symbol, diff, lambdify
sym.init_printing()

# For Getting Access to Another Directory
import os

# For measuring computation time
import time



class PrincipalCoordSystem():
    """ Define Principal Coordinate System """
    
    def s(self, x):
        """ Coordintae Transformation """
        s = np.ndarray((2,), 'object')
        
        """ 2nd Order Polynomial """
        s[0] = x[0]**2 - x[1]**2
        s[1] = 2*x[0]*x[1]
        """ 3rd Order Polynomial """
#        s[0] = x[0]**3 - 3*x[0]*x[1]**2
#        s[1] = -x[1]**3 + 3*x[0]**2*x[1]
        """ 4th Order Polynomial """
#        s[0] = x[0]**4 - 6*x[0]**2*x[1]**2 + x[1]**4
#        s[1] = 4*x[0]**3*x[1] - 4*x[0]*x[1]**3
        
        return s

    def u(self, s):
        """ Target Function under New Coordinate System """
        u = s[0]
        
        return u
    
    
class Theory(PrincipalCoordSystem):
    """  Theoretical Values of Taylor Series Coefficients """
    
    def __init__(self, x, s, x_value):
        self.PCS = PrincipalCoordSystem()
        self.x = x
        self.s = s
        self.x_value = x_value
    
    def s_theory(self):
        """  x_Taylor Series Coefficients of s1 """
        x_value = self.x_value
        s_theory = self.PCS.s(x_value)
        
        return s_theory
    
    def a_theory(self):
        """  x_Taylor Series Coefficients of s1 """
        x = self.x
        s1 = self.PCS.s(x)[0]
        x_value = self.x_value
        
        a = np.ndarray((6,), 'object')
        a[0] = s1
        a[1] = diff(s1, x[0])
        a[2] = diff(s1, x[1])
        a[3] = diff(s1, x[0], 2)
        a[4] = diff(s1, x[0], x[1])
        a[5] = diff(s1, x[1], 2)
        
        a_theory = np.ndarray((len(a),), 'object')
        for i in range(len(a)):
            a_theory[i] = lambdify(x, a[i], 'numpy')
            a_theory[i] = a_theory[i](x_value[0], x_value[1])
        
        return a_theory
    
    def b_theory(self):
        """  x_Taylor Series Coefficients of s2 """
        x = self.x
        s2 = self.PCS.s(x)[1]
        x_value = self.x_value
        
        b = np.ndarray((6,), 'object')
        b[0] = s2
        b[1] = diff(s2, x[0])
        b[2] = diff(s2, x[1])
        b[3] = diff(s2, x[0], 2)
        b[4] = diff(s2, x[0], x[1])
        b[5] = diff(s2, x[1], 2)
        
        b_theory = np.ndarray((len(b),), 'object')
        for i in range(len(b)):
            b_theory[i] = lambdify(x, b[i], 'numpy')
            b_theory[i] = b_theory[i](x_value[0], x_value[1])
            
        return b_theory

    def r_theory(self):
        """ s_Taylor Series Coefficients of u """
        s = self.s
        u = self.PCS.u(s)
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
        
    
class Plot(PrincipalCoordSystem):
    """ Display Plot """
    
    def __init__(self, x_value):
        self.PCS = PrincipalCoordSystem()
        self.x_value = x_value
        
    def u_plot(self):
        x_value = self.x_value
        s = self.PCS.s(x_value)
        u = self.PCS.u(s)
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_wireframe(x_value[0], x_value[1], u, linewidth = 0.2)

        plt.savefig('target_function_3d.pdf')
        plt.savefig('target_function_3d.png')
        plt.pause(.01)
       
    def principal_coordinate_system_plot(self):
        x_value = self.x_value
        s = self.PCS.s(x_value)
            
        plt.gca().set_aspect('equal', adjustable='box')
        
        interval1 = np.arange(-100, 100, 1.0)
        interval2 = np.arange(-100, 100, 1.0)
        
        cr_s1 = plt.contour(x_value[0], x_value[1], s[0], interval1, colors = 'red')
#        levels1 = cr_s1.levels
#        cr_s1.clabel(levels1[::5], fmt = '%3.1f')
        
        cr_s2 = plt.contour(x_value[0], x_value[1], s[1], interval2, colors = 'blue')
#        levels2 = cr_s2.levels
#        cr_s2.clabel(levels2[::5], fmt = '%3.1f')
        
        plt.savefig('principal_coordinate_system.pdf')
        plt.savefig('principal_coordinate_system.png')
        plt.pause(.01)
        


if __name__ == '__main__':
    
    t0 = time.time()
    
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    n = 5
    
    x_value = np.ndarray((len(x),))
    s_value = np.ndarray((len(s),))
    
    x_value_array = np.ndarray((n + 1, n + 1, len(x),))
    s_theory_array = np.ndarray((n + 1, n + 1, len(s),))
    a_theory_array = np.ndarray((n + 1, n + 1, 6,))
    b_theory_array = np.ndarray((n + 1, n + 1, 6,))
    r_theory_array = np.ndarray((n + 1, n + 1, 6,))
    
    ###############################
    Theory = Theory(x, s, x_value)
    ###############################  
    
    for i in range(n + 1):
        for j in range(n + 1):
            x_value[0] = 1.0 + i/n
            x_value[1] = 1.0 + j/n
        
            s_theory = Theory.s_theory()
            a_theory = Theory.a_theory()
            b_theory = Theory.b_theory()
            r_theory = Theory.r_theory()
            
            for k in range(len(x)):
                x_value_array[i][j][k] = x_value[k]
                
            for k in range(len(s)):
                s_theory_array[i][j][k] = s_theory[k]
            
            for k in range(6):
                a_theory_array[i][j][k] = a_theory[k]
                
            for k in range(6):
                b_theory_array[i][j][k] = b_theory[k]
                
            for k in range(6):
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

    x_value = np.meshgrid(np.arange(0, 2, 0.01), 
                          np.arange(0, 2, 0.01))
    
    #####################
    Plot = Plot(x_value)
    #####################
    os.chdir('./graph')
    
    print('3D Plot of u')
    Plot.u_plot()
    print('')
    
    print('Principal Coordinate System')
    Plot.principal_coordinate_system_plot()
    print('')    
    
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                                                                                                    