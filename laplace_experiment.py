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

# For Symbolic Notation
import sympy as sym
from sympy import Symbol, diff, nsolve, lambdify
sym.init_printing()

# For Symbolic Expression Displaying
from IPython.display import display

# For Random Variables
import random

# For measuring computation time
import time



class Experiment(laplace_theory.Theory):
    """ Tayolr Series Expression of Parameters """
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    def __init__(self, s, x):
        self.theory = laplace_theory.Theory()
        self.u_theory = self.theory.u(x)
        self.s_values = self.theory.s_values()
        self.x_values = self.theory.x_values(x)
        self.r_theory = self.theory.r_theory(x)
        self.a_theory = self.theory.a_theory(x)
        self.b_theory = self.theory.b_theory(x)
        
        known = np.ndarray((12,))
        known[0] = r_theory[0]
        known[1] = r_theory[1]
        known[2] = r_theory[2]
        known[3] = r_theory[3]
        known[4] = r_theory[4]
        known[5] = r_theory[5]
        known[6] = a_theory[0]
        known[7] = a_theory[1]
        known[8] = a_theory[2]
        known[9] = b_theory[0]
        known[10] = b_theory[1]
        known[11] = b_theory[2]
        self.known = known
        
        unknown = np.ndarray((6,), 'object')
        unknown[0] = Symbol('a11', real = True)
        unknown[1] = Symbol('a12', real = True)
        unknown[2] = Symbol('a22', real = True)
        unknown[3] = Symbol('b11', real = True)
        unknown[4] = Symbol('b12', real = True)
        unknown[5] = Symbol('b22', real = True)
        self.unknown = unknown

    def s_taylor_u(self, s):
        """ 2nd Order s_Taylor Series of u """
        known = self.known
        s_value = self.s_values[0][0]
        
        return known[0] \
               + known[1]*(s[0] - s_value[0]) \
               + known[2]*(s[1] - s_value[1]) \
               + known[3]*(s[0] - s_value[0])**2/2 \
               + known[4]*(s[0] - s_value[0])*(s[1] - s_value[1]) \
               + known[5]*(s[1] - s_value[1])**2/2
    
    def x_taylor_s1(self, x):
        """ 2nd Order x_Taylor Series of s1 """
        known = self.known
        unknown = self.unknown
        x_value = self.x_values[0][0]
        
        return known[6] \
               + known[7]*(x[0] - x_value[0]) \
               + known[8]*(x[1] - x_value[1]) \
               + unknown[0]*(x[0] - x_value[0])**2/2 \
               + unknown[1]*(x[0] - x_value[0])*(x[1] - x_value[1]) \
               + unknown[2]*(x[1] - x_value[1])**2/2
        
    def x_taylor_s2(self, x):
        """ 2nd Order x_Taylor Series of s2 """
        known = self.known
        unknown = self.unknown
        x_value = self.x_values[0][0]
        
        return known[9] \
               + known[10]*(x[0] - x_value[0]) \
               + known[11]*(x[1] - x_value[1]) \
               + unknown[3]*(x[0] - x_value[0])**2/2 \
               + unknown[4]*(x[0] - x_value[0])*(x[1] - x_value[1]) \
               + unknown[5]*(x[1] - x_value[1])**2/2
               
    def x_taylor_u(self, s, x):
        """ 4th Order x_taylor Series of u"""
        u = self.s_taylor_u(s)
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        
        u = lambdify(s, u, 'numpy')
        u = u(s1, s2)
        
        return u
    
    def s_taylor_du_ds(self, s):
        """ 1st Order s_Taylor Series of (du/ds) """
        u = self.s_taylor_u(s)
        
        du_ds1 = diff(u, s[0])
        du_ds2 = diff(u, s[1])
        du_ds = np.ndarray((2,), 'object')
        du_ds[0] = du_ds1
        du_ds[1] = du_ds2
        
        return du_ds
                
    def x_taylor_du_ds(self, s, x):
        """ 1st Order x_Taylor Series of (du/ds) """
        du_ds1 = self.s_taylor_du_ds(s)[0]
        du_ds2 = self.s_taylor_du_ds(s)[1]
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        
        du_ds1 = lambdify(s, du_ds1, 'numpy')
        du_ds2 = lambdify(s, du_ds2, 'numpy')
        du_ds1 = du_ds1(s1, s2)
        du_ds2 = du_ds2(s1, s2)
        du_ds = np.ndarray((2,), 'object')
        du_ds[0] = du_ds1
        du_ds[1] = du_ds2
        
        return du_ds
        
    def x_taylor_ddu_dds(self, s, x):
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
                ddu_dds[i, j] = lambdify(s, ddu_dds[i, j], 'numpy')
                ddu_dds[i, j] = ddu_dds[i, j](s1, s2)       
        ddu_dds1 = ddu_dds[0][0]
        ddu_ds1ds2 = ddu_dds[0][1]
        ddu_ds2ds1 = ddu_dds[1][0]
        ddu_dds2 = ddu_dds[1][1]
        ddu_dds = np.ndarray((2, 2,), 'object')
        ddu_dds[0][0] = ddu_dds1
        ddu_dds[0][1] = ddu_ds1ds2
        ddu_dds[1][0] = ddu_ds2ds1
        ddu_dds[1][1] = ddu_dds2        
        
        return ddu_dds
       
    def x_taylor_ds_dx(self, x):
        """ 1st Order x_Taylor Series of (ds/dx) """
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        
        ds1_dx1 = diff(s1, x[0])
        ds1_dx2 = diff(s1, x[1])
        ds2_dx1 = diff(s2, x[0])
        ds2_dx2 = diff(s2, x[1])
        ds_dx = np.ndarray((2, 2,), 'object')
        ds_dx[0][0] = ds1_dx1
        ds_dx[0][1] = ds1_dx2
        ds_dx[1][0] = ds2_dx1
        ds_dx[1][1] = ds2_dx2
        
        return ds_dx
        
    def x_taylor_submetric(self, x):
        """ 2nd Order x_Taylor Series of Subscript Metric g_ij """
        ds_dx = self.x_taylor_ds_dx(x)
        
        ds_dx1 = [ds_dx[0][0], 
                  ds_dx[1][0]]
        ds_dx2 = [ds_dx[0][1], 
                  ds_dx[1][1]]
        g11 = np.dot(ds_dx1, ds_dx1)
        g12 = np.dot(ds_dx1, ds_dx2)
        g21 = np.dot(ds_dx2, ds_dx1)
        g22 = np.dot(ds_dx2, ds_dx2)
        submetric = np.ndarray((2, 2,), 'object')
        submetric[0][0] = g11
        submetric[0][1] = g12
        submetric[1][0] = g21
        submetric[1][1] = g22
        
        return submetric
        
    def term_linear_x_taylor_g12(self, x):
        x_value = self.x_values[0][0]
        g12 = self.x_taylor_submetric(x)[0][1]
        
        coeff_g12 = np.ndarray((len(x_value) + 1,), 'object')
        coeff_g12[0] = diff(g12, x[0])
        coeff_g12[1] = diff(g12, x[1])
        coeff_g12[2] = g12 \
                       - diff(g12, x[0])*x[0] \
                       - diff(g12, x[1])*x[1]
        for i in range(len(x_value) + 1):
            coeff_g12[i] = lambdify(x, coeff_g12[i], 'numpy')
            coeff_g12[i] = coeff_g12[i](x_value[0], x_value[1]) 
         
        return coeff_g12
    
    
    def linear_x_taylor_dx_ds(self, x):
        """ 1st Order x_Taylor Series of (dx/ds) """
        """ NOT Using Inverse Matrix Library to Reduce Comutational Cost"""
        x_value = self.x_values[0][0]
        ds_dx = self.x_taylor_ds_dx(x)
        
        det = ds_dx[0][0]*ds_dx[1][1] - ds_dx[0][1]*ds_dx[1][0]
        linear_dx_ds = np.ndarray((2, 2), 'object')
        for i in range(2):
            for j in range(2):
                coeff_dx_ds = np.ndarray((3,), 'object')
                coeff_dx_ds[0] = ds_dx[i][j]/det \
                                 *(diff(ds_dx[i][j], x[0])/ds_dx[i][j] \
                                   - diff(det, x[0])/det)
                coeff_dx_ds[1] = ds_dx[i][j]/det \
                                 *(diff(ds_dx[i][j], x[1])/ds_dx[i][j] \
                                   - diff(det, x[1]))
                coeff_dx_ds[2] = ds_dx[i, j]/det
                for k in range(len(x_value) + 1):
                    coeff_dx_ds[k] = lambdify(x, coeff_dx_ds[k], 'numpy')
                    coeff_dx_ds[k] = coeff_dx_ds[k](x_value[0], x_value[1])
                linear_dx_ds[i][j] = coeff_dx_ds[0]*(x[0] - x_value[0]) \
                                     + coeff_dx_ds[1]*(x[1] - x_value[1]) \
                                     + coeff_dx_ds[2] 
        dx1_ds1 = linear_dx_ds[1][1]
        dx1_ds2 = - linear_dx_ds[0][1]
        dx2_ds1 = - linear_dx_ds[1][0]
        dx2_ds2 = linear_dx_ds[0][0]
        dx_ds = np.ndarray((2, 2,), 'object')
        dx_ds[0][0] = dx1_ds1
        dx_ds[0][1] = dx1_ds2
        dx_ds[1][0] = dx2_ds1
        dx_ds[1][1] = dx2_ds2
        
        return dx_ds
  
    def modified_x_taylor_dg_ds1(self, x):
        """ 2nd Order Modified x_Taylor Series of dg11/ds1 """
        """ dg_ij/ds1 = dx1/ds1*dg_ij/dx1 + dx2/ds1*dg_ij/dx2 """
        linear_dx1_ds1 = self.linear_x_taylor_dx_ds(x)[0][0]
        linear_dx2_ds1 = self.linear_x_taylor_dx_ds(x)[1][0]
        g11 = self.x_taylor_submetric(x)[0][0]
        g12 = self.x_taylor_submetric(x)[0][1]
        g21 = self.x_taylor_submetric(x)[1][0]
        g22 = self.x_taylor_submetric(x)[1][1]
        
        dg11_dx1 = diff(g11, x[0])
        dg11_dx2 = diff(g11, x[1])
        dg12_dx1 = diff(g12, x[0])
        dg12_dx2 = diff(g12, x[1])
        dg21_dx1 = diff(g21, x[0])
        dg21_dx2 = diff(g21, x[1])
        dg22_dx1 = diff(g22, x[0])
        dg22_dx2 = diff(g22, x[1])
        
        dg11_ds1 = linear_dx1_ds1*dg11_dx1 + linear_dx2_ds1*dg11_dx2
        dg12_ds1 = linear_dx1_ds1*dg12_dx1 + linear_dx2_ds1*dg12_dx2
        dg21_ds1 = linear_dx1_ds1*dg21_dx1 + linear_dx2_ds1*dg21_dx2
        dg22_ds1 = linear_dx1_ds1*dg22_dx1 + linear_dx2_ds1*dg22_dx2
        
        dg_ds1 = np.ndarray((2, 2,), 'object')
        dg_ds1[0][0] = dg11_ds1
        dg_ds1[0][1] = dg12_ds1
        dg_ds1[1][0] = dg21_ds1
        dg_ds1[1][1] = dg22_ds1
        
        return dg_ds1
    
    def term_linear_x_taylor_laplacian_u(self, s, x):
        """ 1st Order x_Taylor Series of Laplacian of u """
        """ 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
        x_value = self.x_values[0][0]
        du_ds1 = self.x_taylor_du_ds(s, x)[0]
        ddu_dds1 = self.x_taylor_ddu_dds(s, x)[0][0]
        g11 = self.x_taylor_submetric(x)[0][0]
        g22 = self.x_taylor_submetric(x)[1][1]
        modified_dg11_ds1 = self.modified_x_taylor_dg_ds1(x)[0][0]
        modified_dg22_ds1 = self.modified_x_taylor_dg_ds1(x)[1][1]
        
        laplacian_u = 2*g11*g22*ddu_dds1 \
                      + (g11*modified_dg22_ds1 - \
                         g22*modified_dg11_ds1)*du_ds1
        
        coeff_laplacian_u = np.ndarray((len(x_value) + 1,), 'object')
        coeff_laplacian_u[0] = diff(laplacian_u, x[0])
        coeff_laplacian_u[1] = diff(laplacian_u, x[1])
        coeff_laplacian_u[2] = laplacian_u \
                               - diff(laplacian_u, x[0])*x[0] \
                               - diff(laplacian_u, x[1])*x[1]
        for i in range(len(x_value) + 1):
            coeff_laplacian_u[i] = lambdify(x, coeff_laplacian_u[i], 'numpy')
            coeff_laplacian_u[i] = coeff_laplacian_u[i](x_value[0], x_value[1]) 
         
        return coeff_laplacian_u
    
    def solution(self, s, x):
        unknown = self.unknown
        a_theory = self.a_theory[0][0]
        b_theory = self.b_theory[0][0]
        linear_g12 = self.term_linear_x_taylor_g12(x)
        linear_laplacian_u = self.term_linear_x_taylor_laplacian_u(s, x)
        
        f = np.ndarray((len(unknown),), 'object')
        f[0] = linear_g12[0]
        f[1] = linear_g12[1]
        f[2] = linear_g12[2]
        f[3] = linear_laplacian_u[0]
        f[4] = linear_laplacian_u[1]
        f[5] = linear_laplacian_u[2]
        
        unknown_init = ((1 + random.uniform(-1.0, 1.0)/100)*a_theory[3],
                        (1 + random.uniform(-1.0, 1.0)/100)*a_theory[4],
                        (1 + random.uniform(-1.0, 1.0)/100)*a_theory[5],
                        (1 + random.uniform(-1.0, 1.0)/100)*b_theory[3],
                        (1 + random.uniform(-1.0, 1.0)/100)*b_theory[4],
                        (1 + random.uniform(-1.0, 1.0)/100)*b_theory[5]
                        )
        
        linear_f = np.ndarray((len(f),), 'object')
        for i in range(len(f)):
            coeff_f = np.ndarray((len(f), len(unknown) + 1,), 'object')
            coeff_f[i][0] = diff(f[i], unknown[0])
            coeff_f[i][1] = diff(f[i], unknown[1])
            coeff_f[i][2] = diff(f[i], unknown[2])
            coeff_f[i][3] = diff(f[i], unknown[3])
            coeff_f[i][4] = diff(f[i], unknown[4])
            coeff_f[i][5] = diff(f[i], unknown[5])
            coeff_f[i][6] = f[i]
            for k in range(len(unknown) + 1):
                coeff_f[i][k] = lambdify(unknown, coeff_f[i][k], 'numpy')
                coeff_f[i][k] = coeff_f[i][k](unknown_init[0],
                                              unknown_init[1],
                                              unknown_init[2],
                                              unknown_init[3],
                                              unknown_init[4],
                                              unknown_init[5]
                                              )
            linear_f[i] = coeff_f[i][0]*(unknown[0] - unknown_init[0]) \
                          + coeff_f[i][1]*(unknown[1] - unknown_init[1]) \
                          + coeff_f[i][2]*(unknown[2] - unknown_init[2]) \
                          + coeff_f[i][3]*(unknown[3] - unknown_init[3]) \
                          + coeff_f[i][4]*(unknown[4] - unknown_init[4]) \
                          + coeff_f[i][5]*(unknown[5] - unknown_init[5]) \
                          + coeff_f[i][6]     
        solution = nsolve(linear_f, unknown, unknown_init)
        
        return solution

        

if __name__ == '__main__':
    
    t0 = time.time()
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)

    unknown = np.ndarray((6,), 'object')
    unknown[0] = Symbol('a11', real = True)
    unknown[1] = Symbol('a12', real = True)
    unknown[2] = Symbol('a22', real = True)
    unknown[3] = Symbol('b11', real = True)
    unknown[4] = Symbol('b12', real = True)
    unknown[5] = Symbol('b22', real = True)
    
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
    
    
    experiment = Experiment(s, x)
    unknown_experiment = experiment.solution(s, x)
    
    print('(a_experiment, b_experiment) = ')
    [print(round(item, 4)) for item in unknown_experiment]
    print('')
    
    
    # Verification
    g12 = experiment.term_linear_x_taylor_g12(x)
    g12 = lambdify(unknown, g12, 'numpy')
    g12 = g12(a_theory[3],
              a_theory[4],
              a_theory[5],
              b_theory[3],
              b_theory[4],
              b_theory[5])
#    g12 = g12(unknown_experiment[0],
#              unknown_experiment[1],
#              unknown_experiment[2],
#              unknown_experiment[3],
#              unknown_experiment[4],
#              unknown_experiment[5])
    print('Verificaiton of g12')
    [print(round(item, 4)) for item in g12]
    print('')
    
    laplacian_u = experiment.term_linear_x_taylor_laplacian_u(s, x)
    laplacian_u = lambdify(unknown, laplacian_u, 'numpy')
    laplacian_u = laplacian_u(a_theory[3],
                              a_theory[4],
                              a_theory[5],
                              b_theory[3],
                              b_theory[4],
                              b_theory[5])
#    laplacian_u = laplacian_u(unknown_experiment[0],
#                              unknown_experiment[1],
#                              unknown_experiment[2],
#                              unknown_experiment[3],
#                              unknown_experiment[4],
#                              unknown_experiment[5])
    print('Verificaiton of Laplacian u')
    [print(round(item, 4)) for item in laplacian_u]
    print('')

    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
        
    










