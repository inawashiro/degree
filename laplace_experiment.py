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
from numpy import dot
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
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    def __init__(self, x, s):
        self.theory = laplace_theory.Theory()
        self.s_values = self.theory.s_values()[0][0]
        self.x_values = self.theory.x_values(x)[0][0]
        self.a_theory = self.theory.a_theory(x)[0][0]
        self.b_theory = self.theory.b_theory(x)[0][0]
        self.r_theory = self.theory.r_theory(x)[0][0]
        
        known = np.ndarray((12,))
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
        self.known = known
        
        unknown = np.ndarray((6,), 'object')
        unknown[0] = Symbol('a11', real = True)
        unknown[1] = Symbol('a12', real = True)
        unknown[2] = Symbol('a22', real = True)
        unknown[3] = Symbol('b11', real = True)
        unknown[4] = Symbol('b12', real = True)
        unknown[5] = Symbol('b22', real = True)
        self.unknown = unknown
    
    def x_taylor_s1(self, x):
        """ 2nd Order x_Taylor Series of s1 """
        known = self.known
        unknown = self.unknown
        x_value = self.x_values
        
        return known[0] \
               + known[1]*(x[0] - x_value[0]) \
               + known[2]*(x[1] - x_value[1]) \
               + unknown[0]*(x[0] - x_value[0])**2/2 \
               + unknown[1]*(x[0] - x_value[0])*(x[1] - x_value[1]) \
               + unknown[2]*(x[1] - x_value[1])**2/2
        
    def x_taylor_s2(self, x):
        """ 2nd Order x_Taylor Series of s2 """
        known = self.known
        unknown = self.unknown
        x_value = self.x_values
        
        return known[3] \
               + known[4]*(x[0] - x_value[0]) \
               + known[5]*(x[1] - x_value[1]) \
               + unknown[3]*(x[0] - x_value[0])**2/2 \
               + unknown[4]*(x[0] - x_value[0])*(x[1] - x_value[1]) \
               + unknown[5]*(x[1] - x_value[1])**2/2
    
    def s_taylor_u(self, s):
        """ 2nd Order s_Taylor Series of u """
        known = self.known
        s_value = self.s_values
        
        return known[6] \
               + known[7]*(s[0] - s_value[0]) \
               + known[8]*(s[1] - s_value[1]) \
               + known[9]*(s[0] - s_value[0])**2/2 \
               + known[10]*(s[0] - s_value[0])*(s[1] - s_value[1]) \
               + known[11]*(s[1] - s_value[1])**2/2
           
#    def x_taylor_u(self, x):
#        """ 4th Order x_taylor Series of u"""
#        u = self.s_taylor_u(s)
#        s1 = self.x_taylor_s1(x)
#        s2 = self.x_taylor_s2(x)
#        
#        u = lambdify(s, u, 'numpy')
#        u = u(s1, s2)
#        
#        return u
    
    def x_taylor_du_ds(self, x):
        """ 1st Order s_Taylor Series of (du/ds) """
        u = self.s_taylor_u(s)
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        
        du_ds = np.ndarray((2,), 'object')
        du_ds[0] = diff(u, s[0])
        du_ds[1] = diff(u, s[1])
        
        for i in range(2):
            du_ds[i] = lambdify(s, du_ds[i], 'numpy')
            du_ds[i] = du_ds[i](s1, s2)
        
        return du_ds
                    
    def x_taylor_ddu_dds(self, x):
        """ 0th Order x_Taylor Series of (ddu/dds) """
        u = self.s_taylor_u(s)
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        
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
       
    def x_taylor_ds_dx(self, x):
        """ 1st Order x_Taylor Series of (ds/dx) """
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        
        ds_dx = np.ndarray((2, 2,), 'object')
        ds_dx[0][0] = diff(s1, x[0])
        ds_dx[0][1] = diff(s1, x[1])
        ds_dx[1][0] = diff(s2, x[0])
        ds_dx[1][1] = diff(s2, x[1])
                
        return ds_dx
        
    def x_taylor_submetric(self, x):
        """ 2nd Order x_Taylor Series of Subscript Metric g_ij """
        ds_dx = self.x_taylor_ds_dx(x)
        
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
    
    def x_taylor_dx_ds(self, x):
        """ 1st Order x_Taylor Series of (dx/ds) """
        """ Inverse Matrix Library NOT Used due to High Computational Cost """
        ds_dx = self.x_taylor_ds_dx(x)
        det = ds_dx[0][0]*ds_dx[1][1] - ds_dx[0][1]*ds_dx[1][0]
        
        dx_ds = np.ndarray((2, 2), 'object')
        dx_ds[0][0] = ds_dx[1][1]/det
        dx_ds[0][1] = - ds_dx[0][1]/det
        dx_ds[1][0] = - ds_dx[1][0]/det
        dx_ds[1][1] = ds_dx[0][0]/det
    
        return dx_ds
  
    def x_taylor_dg_ds1(self, x):
        """ 2nd Order Modified x_Taylor Series of dg11/ds1 """
        """ dg_ij/ds1 = dx1/ds1*dg_ij/dx1 + dx2/ds1*dg_ij/dx2 """
        dx1_ds1 = self.x_taylor_dx_ds(x)[0][0]
        dx2_ds1 = self.x_taylor_dx_ds(x)[1][0]
        submetric = self.x_taylor_submetric(x)
        
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
    def __init__(self):
        self.taylor = Taylor(x, s)
    
    def term_linear_x_taylor_g12(self, x):
        """ 1st Order x_Taylor Series of g_12 """
        x_value = self.taylor.x_values
        g12 = self.taylor.x_taylor_submetric(x)[0][1]
        
        coeff_g12 = np.ndarray((len(x_value) + 1,), 'object')
        coeff_g12[0] = diff(g12, x[0])
        coeff_g12[1] = diff(g12, x[1])
        coeff_g12[2] = g12
        
        for i in range(len(x_value) + 1):
            coeff_g12[i] = lambdify(x, coeff_g12[i], 'numpy')
            coeff_g12[i] = coeff_g12[i](0, 0)
         
        return coeff_g12
    
    def term_linear_x_taylor_laplacian_u(self, x):
        """ 1st Order x_Taylor Series of Laplacian of u """
        """ 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
        x_value = self.taylor.x_values
        du_ds1 = self.taylor.x_taylor_du_ds(x)[0]
        ddu_dds1 = self.taylor.x_taylor_ddu_dds(x)[0][0]
        g11 = self.taylor.x_taylor_submetric(x)[0][0]
        g22 = self.taylor.x_taylor_submetric(x)[1][1]
        dg11_ds1 = self.taylor.x_taylor_dg_ds1(x)[0][0]
        dg22_ds1 = self.taylor.x_taylor_dg_ds1(x)[1][1]
        
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
        g12 = self.term_linear_x_taylor_g12(x)
        laplacian_u = self.term_linear_x_taylor_laplacian_u(x)
        
        f = np.ndarray((len(unknown),), 'object')
        f[0] = g12[0]
        f[1] = g12[1]
        f[2] = g12[2]
        f[3] = laplacian_u[0]
        f[4] = laplacian_u[1]
        f[5] = laplacian_u[2]
        
        return f
    
    def unknown_init(self):
        a_theory = self.taylor.a_theory
        b_theory = self.taylor.b_theory
        
        unknown_init = ((1 + random.uniform(-0.0, 0.0)/100)*a_theory[3],
                        (1 + random.uniform(-0.0, 0.0)/100)*a_theory[4],
                        (1 + random.uniform(-0.0, 0.0)/100)*a_theory[5],
                        (1 + random.uniform(-0.0, 0.0)/100)*b_theory[3],
                        (1 + random.uniform(-0.0, 0.0)/100)*b_theory[4],
                        (1 + random.uniform(-0.0, 0.0)/100)*b_theory[5]
                        )
        
        return unknown_init
    
    def A(self, solution):
        unknown = self.taylor.unknown
        f = self.f()
        
        A = np.ndarray((len(f), len(unknown),), 'object')
        for i in range(len(unknown)):
            for j in range(len(unknown)):
                A[i][j] = diff(f[i], unknown[j])
                A[i][j] = lambdify(unknown, A[i][j], 'numpy')
                A[i][j] = A[i][j](solution[0],
                                  solution[1],
                                  solution[2],
                                  solution[3],
                                  solution[4],
                                  solution[5]
                                  )
        A = A.astype('float')
        
        return A
    
    def b(self, solution):
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
                b[i] = b[i](solution[0],
                            solution[1],
                            solution[2],
                            solution[3],
                            solution[4],
                            solution[5]
                            )            
        b = b.astype('float')
    
        return b    
    
    def solution(self):
        unknown = self.taylor.unknown
        f = self.f()
        solution = self.unknown_init()
        
        def error_norm(solution):
            error = np.ndarray((len(f),), 'object')
            for i in range(len(f)):
                error[i] = lambdify(unknown, f[i], 'numpy')
                error[i] = error[i](solution[0],
                                    solution[1],
                                    solution[2],
                                    solution[3],
                                    solution[4],
                                    solution[5])
            error_norm = norm(error)
            
            return error_norm
        
        error = error_norm(solution)
        
        while error > 1.0e-4:
            A = self.A(solution)
            b = self.b(solution)
            solution = lstsq(A, b)[0]
#            solution = solve(A, b)        
            error = error_norm(solution)
        
        return solution

        

if __name__ == '__main__':
    
    t0 = time.time()
    
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    theory = laplace_theory.Theory()
    x_values = theory.x_values(x)[0][0]
    r_theory = theory.r_theory(x)[0][0]
    a_theory = theory.a_theory(x)[0][0]
    b_theory = theory.b_theory(x)[0][0]
    
    print('x_values = ')
    print(x_values)
    print('')
    
    unknown_theory = np.ndarray((6,))
    unknown_theory[0] = a_theory[3]
    unknown_theory[1] = a_theory[4]
    unknown_theory[2] = a_theory[5]
    unknown_theory[3] = b_theory[3]
    unknown_theory[4] = b_theory[4]
    unknown_theory[5] = b_theory[5]
    print('(a_theory, b_theory) = ')
    [print(round(item, 4)) for item in unknown_theory]
    print('')
    
    
    experiment = Experiment()
    
    def relative_error_norm(a, b):
        relative_error_norm = norm(b - a)/norm(a)
        
        return relative_error_norm
        
    unknown_init = experiment.unknown_init()
    print('(a_init, b_init) = ')
    [print(round(item, 4)) for item in unknown_init]
    print('')
    
    print('Error of (a_init, b_init) = ')
    error_init = relative_error_norm(unknown_theory, unknown_init)
    print(round(error_init, 4)*100, '(%)')
    print('')
    
    unknown_experiment = experiment.solution()
    print('(a_experiment, b_experiment) = ')
    [print(round(item, 4)) for item in unknown_experiment]
    print('')
    
    print('Error of (a_experimrnt, b_experiment) = ')
    error_experiment = relative_error_norm(unknown_theory, unknown_experiment)
    print(round(error_experiment, 4)*100, '(%)')
    print('')
    
    A_init = experiment.A(unknown_init)
    eig_A_init = eig(A_init)[0]
    print('Eigen Vaule of A_init = ')
    [print(item) for item in eig_A_init]
    print('')


    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
        
    





















