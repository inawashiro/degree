# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:00:27 2018

@author: inawashiro
"""
import laplace_theory


# For Visualization
import matplotlib.pyplot as plt

# For Numerical Computation 
import numpy as np
from numpy import dot, absolute
from numpy.linalg import norm, lstsq, solve, eig

# For Symbolic Notation
import sympy as sym
from sympy import Symbol, diff, lambdify
sym.init_printing()

# For Symbolic Expression Displaying
from IPython.display import display

# For Random Variables
import random

# For Measuring Computation Time
import time



class Taylor(laplace_theory.Theory):
    """ Taylor Series Expressions of Parameters"""
    
    def __init__(self, known, x_value, s_value):
        self.known = known
        
        unknown = np.ndarray((6,), 'object')
        unknown[0] = Symbol('a11', real = True)
        unknown[1] = Symbol('a12', real = True)
        unknown[2] = Symbol('a22', real = True)
        unknown[3] = Symbol('b11', real = True)
        unknown[4] = Symbol('b12', real = True)
        unknown[5] = Symbol('b22', real = True)
        self.unknown = unknown
        
        x = np.ndarray((2,), 'object')
        x[0] = Symbol('x1', real = True)
        x[1] = Symbol('x2', real = True)
        self.x = x
    
        s = np.ndarray((2,), 'object')
        s[0] = Symbol('s1', real = True)
        s[1] = Symbol('s2', real = True)
        self.s = s
        
        self.x_value = x_value
        
        self.s_value = s_value
        
    def x_taylor_s1(self):
        """ 2nd Order x_Taylor Series of s1 """
        known = self.known
        unknown = self.unknown
        x = self.x
        x_value = self.x_value
        
        return known[0] \
               + known[1]*(x[0] - x_value[0]) \
               + known[2]*(x[1] - x_value[1]) \
               + unknown[0]*(x[0] - x_value[0])**2/2 \
               + unknown[1]*(x[0] - x_value[0])*(x[1] - x_value[1]) \
               + unknown[2]*(x[1] - x_value[1])**2/2
        
    def x_taylor_s2(self):
        """ 2nd Order x_Taylor Series of s2 """
        known = self.known
        unknown = self.unknown
        x = self.x
        x_value = self.x_value
        
        return known[3] \
               + known[4]*(x[0] - x_value[0]) \
               + known[5]*(x[1] - x_value[1]) \
               + unknown[3]*(x[0] - x_value[0])**2/2 \
               + unknown[4]*(x[0] - x_value[0])*(x[1] - x_value[1]) \
               + unknown[5]*(x[1] - x_value[1])**2/2
    
    def s_taylor_u(self):
        """ 2nd Order s_Taylor Series of u """
        known = self.known
        s = self.s
        s_value = self.s_value
        
        return known[6] \
               + known[7]*(s[0] - s_value[0]) \
               + known[8]*(s[1] - s_value[1]) \
               + known[9]*(s[0] - s_value[0])**2/2 \
               + known[10]*(s[0] - s_value[0])*(s[1] - s_value[1]) \
               + known[11]*(s[1] - s_value[1])**2/2
           
#    def x_taylor_u(self):
#        """ 4th Order x_taylor Series of u"""
#        u = self.s_taylor_u()
#        s1 = self.x_taylor_s1()
#        s2 = self.x_taylor_s2()
#        
#        u = lambdify(s, u, 'numpy')
#        u = u(s1, s2)
#        
#        return u
    
    def x_taylor_du_ds(self):
        """ 1st Order s_Taylor Series of (du/ds) """
        s = self.s
        u = self.s_taylor_u()
        s1 = self.x_taylor_s1()
        s2 = self.x_taylor_s2()
        
        du_ds = np.ndarray((2,), 'object')
        du_ds[0] = diff(u, s[0])
        du_ds[1] = diff(u, s[1])
        
        for i in range(2):
            du_ds[i] = lambdify(s, du_ds[i], 'numpy')
            du_ds[i] = du_ds[i](s1, s2)
        
        return du_ds
                    
    def x_taylor_ddu_dds(self):
        """ 0th Order x_Taylor Series of (ddu/dds) """
        s = self.s
        u = self.s_taylor_u()
        s1 = self.x_taylor_s1()
        s2 = self.x_taylor_s2()
        
        ddu_dds = np.ndarray((2, 2), 'object')
        ddu_dds[0][0] = diff(u, s[0], 2)
        ddu_dds[0][1] = diff(u, s[0], s[1])
        ddu_dds[1][0] = diff(u, s[1], s[0])
        ddu_dds[1][1] = diff(u, s[1], 2)
        
        for i in range(2):
            for j in range(2):
                ddu_dds[i][j] = lambdify(s, ddu_dds[i][j], 'numpy')
                ddu_dds[i][j] = ddu_dds[i][j](s1, s2)       
        
        return ddu_dds
       
    def x_taylor_ds_dx(self):
        """ 1st Order x_Taylor Series of (ds/dx) """
        x = self.x
        s1 = self.x_taylor_s1()
        s2 = self.x_taylor_s2()
        
        ds_dx = np.ndarray((2, 2,), 'object')
        ds_dx[0][0] = diff(s1, x[0])
        ds_dx[0][1] = diff(s1, x[1])
        ds_dx[1][0] = diff(s2, x[0])
        ds_dx[1][1] = diff(s2, x[1])
                
        return ds_dx
        
    def x_taylor_submetric(self):
        """ 2nd Order x_Taylor Series of Subscript Metric g_ij """
        ds_dx = self.x_taylor_ds_dx()
        
        ds_dx1 = np.ndarray((2,), 'object')
        ds_dx1[0] = ds_dx[0][0]
        ds_dx1[1] = ds_dx[1][0]
        ds_dx2 = np.ndarray((2,), 'object')
        ds_dx2[0] = ds_dx[0][1]
        ds_dx2[1] = ds_dx[1][1]
        
        submetric = np.ndarray((2, 2,), 'object')
        submetric[0][0] = dot(ds_dx1, ds_dx1)
        submetric[0][1] = dot(ds_dx1, ds_dx2)
        submetric[1][0] = dot(ds_dx2, ds_dx1)
        submetric[1][1] = dot(ds_dx2, ds_dx2)
        
        return submetric
    
    def x_taylor_dx_ds(self):
        """ 1st Order x_Taylor Series of (dx/ds) """
        """ Inverse Matrix Library NOT Used due to High Computational Cost """
        ds_dx = self.x_taylor_ds_dx()
        det = ds_dx[0][0]*ds_dx[1][1] - ds_dx[0][1]*ds_dx[1][0]
        
        dx_ds = np.ndarray((2, 2), 'object')
        dx_ds[0][0] = ds_dx[1][1]/det
        dx_ds[0][1] = - ds_dx[0][1]/det
        dx_ds[1][0] = - ds_dx[1][0]/det
        dx_ds[1][1] = ds_dx[0][0]/det
    
        return dx_ds
  
    def x_taylor_dg_ds1(self):
        """ 2nd Order Modified x_Taylor Series of dg11/ds1 """
        """ dg_ij/ds1 = dx1/ds1*dg_ij/dx1 + dx2/ds1*dg_ij/dx2 """
        x = self.x
        dx1_ds1 = self.x_taylor_dx_ds()[0][0]
        dx2_ds1 = self.x_taylor_dx_ds()[1][0]
        submetric = self.x_taylor_submetric()
        
        dg_dx1 = np.ndarray((2, 2), 'object')
        for i in range(2):
            for j in range(2):
                dg_dx1[i][j] = diff(submetric[i][j], x[0]) 
                
        dg_dx2 = np.ndarray((2, 2), 'object')
        for i in range(2):
            for j in range(2):
                dg_dx2[i][j] = diff(submetric[i][j], x[1])
                    
        dg_ds1 = np.ndarray((2, 2,), 'object')
        for i in range(2):
            for j in range(2):
                dg_ds1[i][j] = dx1_ds1*dg_dx1[i][j] \
                               + dx2_ds1*dg_dx2[i][j]
        
        return dg_ds1
    
    
class Experiment(Taylor):
    """ Solve Equations """
    def __init__(self, known, x_value, s_value):
        self.taylor = Taylor(known, x_value, s_value)
        
    def term_linear_x_taylor_g12(self):
        """ 1st Order x_Taylor Series of g_12 """
        x = self.taylor.x
        g12 = self.taylor.x_taylor_submetric()[0][1]
        
        coeff_g12 = np.ndarray((len(x_value) + 1,), 'object')
        coeff_g12[0] = diff(g12, x[0])
        coeff_g12[1] = diff(g12, x[1])
        coeff_g12[2] = g12
        
        for i in range(len(x_value) + 1):
            coeff_g12[i] = lambdify(x, coeff_g12[i], 'numpy')
            coeff_g12[i] = coeff_g12[i](0, 0)
         
        return coeff_g12
    
    def term_linear_x_taylor_laplacian_u(self):
        """ 1st Order x_Taylor Series of Laplacian of u """
        """ 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
        x = self.taylor.x
        du_ds1 = self.taylor.x_taylor_du_ds()[0]
        ddu_dds1 = self.taylor.x_taylor_ddu_dds()[0][0]
        g11 = self.taylor.x_taylor_submetric()[0][0]
        g22 = self.taylor.x_taylor_submetric()[1][1]
        dg11_ds1 = self.taylor.x_taylor_dg_ds1()[0][0]
        dg22_ds1 = self.taylor.x_taylor_dg_ds1()[1][1]
        
        laplacian_u = 2*g11*g22*ddu_dds1 \
                      + (g11*dg22_ds1 - g22*dg11_ds1)*du_ds1
        
        coeff_laplacian_u = np.ndarray((len(x_value) + 1,), 'object')
        coeff_laplacian_u[0] = diff(laplacian_u, x[0])
        coeff_laplacian_u[1] = diff(laplacian_u, x[1])
        coeff_laplacian_u[2] = laplacian_u 
        
        for i in range(len(x_value) + 1):
            coeff_laplacian_u[i] = lambdify(x, coeff_laplacian_u[i], 'numpy')
            coeff_laplacian_u[i] = coeff_laplacian_u[i](0, 0)
         
        return coeff_laplacian_u
    
    def f(self):
        unknown = self.taylor.unknown
        g12 = self.term_linear_x_taylor_g12()
        laplacian_u = self.term_linear_x_taylor_laplacian_u()
        
        f = np.ndarray((len(unknown),), 'object')
        f[0] = g12[0]
        f[1] = g12[1]
        f[2] = g12[2]
        f[3] = laplacian_u[0]
        f[4] = laplacian_u[1]
        f[5] = laplacian_u[2]
        
        return f
    
    def unknown_init(self, unknown_theory):
        unknown_init = ((1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[0],
                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[1],
                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[2],
                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[3],
                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[4],
                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[5]
                        )
        
        return unknown_init
    
    def A(self, unknown_temp):
        unknown = self.taylor.unknown
        f = self.f()
        
        A = np.ndarray((len(f), len(unknown),), 'object')
        for i in range(len(unknown)):
            for j in range(len(unknown)):
                A[i][j] = diff(f[i], unknown[j])
                A[i][j] = lambdify(unknown, A[i][j], 'numpy')
                A[i][j] = A[i][j](unknown_temp[0],
                                  unknown_temp[1],
                                  unknown_temp[2],
                                  unknown_temp[3],
                                  unknown_temp[4],
                                  unknown_temp[5]
                                  )
        A = A.astype('float')
        
        return A
    
    def b(self, unknown_temp):
        unknown = self.taylor.unknown
        f = self.f()
        
        b = np.ndarray((len(f),), 'object')
        for i in range(len(f)):
                b[i] = - f[i] \
                       + diff(f[i], unknown[0])*unknown[0] \
                       + diff(f[i], unknown[1])*unknown[1] \
                       + diff(f[i], unknown[2])*unknown[2] \
                       + diff(f[i], unknown[3])*unknown[3] \
                       + diff(f[i], unknown[4])*unknown[4] \
                       + diff(f[i], unknown[5])*unknown[5]
                b[i] = lambdify(unknown, b[i], 'numpy')
                b[i] = b[i](unknown_temp[0],
                            unknown_temp[1],
                            unknown_temp[2],
                            unknown_temp[3],
                            unknown_temp[4],
                            unknown_temp[5]
                            )            
        b = b.astype('float')
    
        return b    
    
    def solution(self, unknown_theory):
        unknown = self.taylor.unknown
        f = self.f()
        unknown_temp = self.unknown_init(unknown_theory)
        
        def error_norm(unknown_temp):
            error = np.ndarray((len(f),), 'object')
            for i in range(len(f)):
                error[i] = lambdify(unknown, f[i], 'numpy')
                error[i] = error[i](unknown_temp[0],
                                    unknown_temp[1],
                                    unknown_temp[2],
                                    unknown_temp[3],
                                    unknown_temp[4],
                                    unknown_temp[5])
            error_norm = norm(error)
            
            return error_norm
        
        error = error_norm(unknown_temp)
        
        while error > 1.0e-4:
            A = self.A(unknown_temp)
            b = self.b(unknown_temp)
            unknown_temp = lstsq(A, b)[0]
#            solution = solve(A, b)        
            error = error_norm(unknown_temp)
        
        solution = unknown_temp
        
        return solution

        

if __name__ == '__main__':
    
    t0 = time.time()
    
    function = laplace_theory.Function()
    
    def relative_error_norm(a, b):
        relative_error_norm = round(norm(b - a)/norm(a), 4)*100
        
        return relative_error_norm
    
    x = function.x
    s = function.s
    a = function.a(x)
    b = function.b(x)
    r = function.r(s)
    
    n = 1
    
    x_value = np.ndarray((len(x),))
    
    x_value_array = np.ndarray((n, n, len(x),))
    s_theory_array = np.ndarray((n, n, len(s)))
    a_theory_array = np.ndarray((n, n, len(a),))
    b_theory_array = np.ndarray((n, n, len(b),))
    r_theory_array = np.ndarray((n, n, len(r),))
    
    known = np.ndarray((12,))
    unknown = np.ndarray(18 - len(known),)
    
    unknown_theory = np.ndarray(len(unknown),)
    unknown_experiment = np.ndarray(len(unknown),)
    
    known_array = np.ndarray((n, n, len(known),))
    unknown_theory_array = np.ndarray((n, n, len(unknown),))
    unknown_init_array = np.ndarray((n, n, len(unknown),))
    unknown_experiment_array = np.ndarray((n, n, len(unknown),))
    
    error_init_array = np.ndarray((n, n, 1))
    error_experiment_array = np.ndarray((n, n, 1))
    
    eig_A_init_array = np.ndarray((n, n, len(unknown),))
    
    
    theory = laplace_theory.Theory()
    
    for i in range(n):
        for j in range(n):
            x_value[0] = 1.0 + i/n
            x_value[1] = 1.0 + j/n
            
            s_theory = theory.s_theory(x_value)
            a_theory = theory.a_theory(x_value)
            b_theory = theory.b_theory(x_value)
            r_theory = theory.r_theory(s_theory)
            
            experiment = Experiment(known, x_value, s_theory)
            
            known[0] = a_theory[0]
            known[1] = a_theory[1]
            known[2] = a_theory[2]
            known[3] = b_theory[0]
            known[4] = b_theory[1]
            known[5] = b_theory[2]
            known[6] = r_theory[0]
            known[7] = r_theory[1]
            known[8] = r_theory[2]
            known[9] = r_theory[3]
            known[10] = r_theory[4]
            known[11] = r_theory[5]
            
            unknown_theory[0] = a_theory[3]
            unknown_theory[1] = a_theory[4]
            unknown_theory[2] = a_theory[5]
            unknown_theory[3] = b_theory[3]
            unknown_theory[4] = b_theory[4]
            unknown_theory[5] = b_theory[5]
            
            unknown_init = experiment.unknown_init(unknown_theory)
            error_init = relative_error_norm(unknown_theory, unknown_init)
            
            unknown_experiment = experiment.solution(unknown_theory)
            error_experiment = relative_error_norm(unknown_theory, unknown_experiment)
            
            A_init = experiment.A(unknown_init)
            eig_A_init = eig(A_init)[0]
            
            for k in range(len(unknown)):
                if absolute(eig_A_init[k]) < 1.0e-4:
                    eig_A_init[k] = 0
                else:
                    real = round(eig_A_init[k].real, 4)
                    imag = round(eig_A_init[k].imag, 4)
                    eig_A_init[k] = real + imag*1j
                
            for k in range(len(x)):
                x_value_array[i][j][k] = x_value[k]
                
            for k in range(len(s)):
                s_theory_array[i][j][k] = s_theory[k]
            
            for k in range(len(a)):
                a_theory_array[i][j][k] = a_theory[k]
            
            for k in range(len(b)):
                b_theory_array[i][j][k] = b_theory[k]
                
            for k in range(len(r)):
                r_theory_array[i][j][k] = r_theory[k]
                
            for k in range(len(known)):
                known_array[i][j][k] = known[k]
                
            for k in range(len(unknown)):
                unknown_theory_array[i][j][k] = unknown_theory[k]
                unknown_init_array[i][j][k] = unknown_init[k]
                unknown_experiment_array[i][j][k] = unknown_experiment[k]
                eig_A_init_array[i][j][k] = eig_A_init[k]
                
            for k in range(1):
                error_init_array[i][j][k] = error_init
                error_experiment_array[i][j][k] = error_experiment
            
    print('x_values = ')
    print(x_value_array)
    print('')

    print('unknown_theory = ')
    print(unknown_theory_array)
    print('')
    
    print('unknown_init = ')
    print(unknown_init_array)
    print('') 
        
    print('Error(%) of unknown_init = ')
    print(error_init_array)
    print('')
    
    unknown_experiment = experiment.solution(unknown_theory)
    print('unknown_experiment = ')
    print(unknown_experiment_array)
    print('')
    
    print('Error(%) of unknown_experiment = ')
    print(error_experiment_array)
    print('')
    
    print('Eigen Vaule of A_init = ')
    print(eig_A_init_array)
    print('')


    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
        
    

























