# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:00:27 2018

@author: inawashiro
"""
import laplace_theory


## For Visualization
#import matplotlib.pyplot as plt
#plt.rcParams['contour.negative_linestyle']='solid'

# For Numerical Computation 
import numpy as np

# For Symbolic Notation
import sympy as sym
from sympy import Symbol, diff, Matrix, simplify, factor, S, Poly, nsolve, lambdify
sym.init_printing()

## For Symbolic Expression Displaying
#from IPython.display import display

# For Random Variables
import random

# For measuring computation time
import time



class Experiment(laplace_theory.Theory):
    """ Tayolr Series Expression of Parameters """
    
    s = [Symbol('s1', real = True), 
         Symbol('s2', real = True)
         ]
    x = [Symbol('x1', real = True), 
         Symbol('x2', real = True)
         ]
    
    def __init__(self, s, x):
        self.theory = laplace_theory.Theory()
        self.u_theory = self.theory.u(x)
        self.s_values = self.theory.s_values()
        self.x_values = self.theory.x_values(x)
        self.r_theory = self.theory.r_theory(x)
        self.a_theory = self.theory.a_theory(x)
        self.b_theory = self.theory.b_theory(x)
        
        self.known = [r_theory[0],
                      r_theory[1],
                      r_theory[2],
                      r_theory[3],
                      r_theory[4],
                      r_theory[5],
                      a_theory[0],
                      a_theory[1],
                      a_theory[2],
                      b_theory[0],
                      b_theory[1],
                      b_theory[2]
                      ]
        self.unknown = [Symbol('a11', real = True),
                        Symbol('a12', real = True),
                        Symbol('a22', real = True),
                        Symbol('b11', real = True),
                        Symbol('b12', real = True),
                        Symbol('b22', real = True)
                        ]

    def s_taylor_u(self, s):
        """ 2nd Order s_Taylor Series of u """
        known = self.known
        s_value = self.s_values[0][0]
        return known[0] \
               + known[1]*(s[0] - s_value[0]) \
               + known[2]*(s[1] - s_value[1]) \
               + known[3]*(s[0] - s_value[0])**2 \
               + known[4]*(s[0] - s_value[0])*(s[1] - s_value[1]) \
               + known[5]*(s[1] - s_value[1])**2
    
    def x_taylor_s1(self, x):
        """ 2nd Order x_Taylor Series of s1 """
        known = self.known
        unknown = self.unknown
        x_value = self.x_values[0][0]
        return known[6] \
               + known[7]*(x[0] - x_value[0]) \
               + known[8]*(x[1] - x_value[1]) \
               + unknown[0]*(x[0] - x_value[0])**2 \
               + unknown[1]*(x[0] - x_value[0])*(x[1] - x_value[1]) \
               + unknown[2]*(x[1] - x_value[1])**2
        
    def x_taylor_s2(self, x):
        """ 2nd Order x_Taylor Series of s1 """
        known = self.known
        unknown = self.unknown
        x_value = self.x_values[0][0]
        return known[9] \
               + known[10]*(x[0] - x_value[0]) \
               + known[11]*(x[1] - x_value[1]) \
               + unknown[3]*(x[0] - x_value[0])**2 \
               + unknown[4]*(x[0] - x_value[0])*(x[1] - x_value[1]) \
               + unknown[5]*(x[1] - x_value[1])**2
               
    def x_taylor_u(self, s, x):
        """ 4th Order x_taylor Series of u"""
        u = self.s_taylor_u(s)
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        u = u.subs([(s[0], s1), (s[1], s2)])
        return u
    
    def s_taylor_du_ds(self, s):
        """ 1st Order s_Taylor Series of (du/ds) """
        u = self.s_taylor_u(s)
        du_ds1 = diff(u, s[0])
        du_ds2 = diff(u, s[1])
        return sym.Matrix([du_ds1,
                           du_ds2])
                
    def x_taylor_du_ds(self, s, x):
        """ 1st Order x_Taylor Series of (du/ds) """
        du_ds1 = self.s_taylor_du_ds(s)[0]
        du_ds2 = self.s_taylor_du_ds(s)[1]
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        du_ds1 = du_ds1.subs([(s[0], s1), (s[1], s2)])
        du_ds2 = du_ds2.subs([(s[0], s1), (s[1], s2)])
        return sym.Matrix([du_ds1,
                           du_ds2
                           ])
        
    def x_taylor_ddu_dds(self, s, x):
        """ 0th Order x_Taylor Series of (ddu/dds) """
        u = self.s_taylor_u(s)
        ddu_dds1 = diff(u, s[0], 2)
        ddu_ds1ds2 = diff(u, s[0], s[1])
        ddu_dds2 = diff(u, s[1], 2)
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        ddu_dds1 = ddu_dds1.subs([(s[0], s1), (s[1], s2)])
        ddu_ds1ds2 = ddu_ds1ds2.subs([(s[0], s1), (s[1], s2)])
        ddu_dds2 = ddu_dds2.subs([(s[0], s1), (s[1], s2)])
        return sym.Matrix([[ddu_dds1, ddu_ds1ds2],
                           [ddu_ds1ds2, ddu_dds2]
                           ])
       
    def x_taylor_ds_dx(self, x):
        """ 1st Order x_Taylor Series of (ds/dx) """
        s1 = self.x_taylor_s1(x)
        s2 = self.x_taylor_s2(x)
        ds1_dx1 = diff(s1, x[0])
        ds1_dx2 = diff(s1, x[1])
        ds2_dx1 = diff(s2, x[0])
        ds2_dx2 = diff(s2, x[1])
        return sym.Matrix([[ds1_dx1, ds1_dx2],
                           [ds2_dx1, ds2_dx2]
                           ])
        
    def x_taylor_submetric(self, x):
        """ 2nd Order x_Taylor Series of Subscript Metric g_ij """
        ds_dx = self.x_taylor_ds_dx(x)
        ds_dx1 = [ds_dx[0, 0], 
                  ds_dx[1, 0]]
        ds_dx2 = [ds_dx[0, 1], 
                  ds_dx[1, 1]]
        g11 = np.dot(ds_dx1, ds_dx1)
        g12 = np.dot(ds_dx1, ds_dx2)
        g21 = np.dot(ds_dx2, ds_dx1)
        g22 = np.dot(ds_dx2, ds_dx2)
        return sym.Matrix([[g11, g12],
                           [g21, g22]
                           ])
        
    def modified_x_taylor_dx_ds(self, x):
        """ 1st Order Modified x_Taylor Series of (dx/ds) """
        """ NOT Using Inverse Matrix Computing Library for Comutational Cost"""
        x_value = self.x_values[0][0]
        ds_dx = self.x_taylor_ds_dx(x)
        
        ds1_dx1 = ds_dx[0, 0]
        ds1_dx2 = ds_dx[0, 1]
        ds2_dx1 = ds_dx[1, 0]
        ds2_dx2 = ds_dx[1, 1]
               
        det = ds1_dx1*ds2_dx2 - ds1_dx2*ds2_dx1

        coeff_0_dx1_ds1 = ds2_dx2/det
        coeff_1_dx1_ds1 = ds2_dx2/det \
                          *(diff(ds2_dx2, x[0])/ds2_dx2 - diff(det, x[0])/det)
        coeff_2_dx1_ds1 = ds2_dx2/det \
                          *(diff(ds2_dx2, x[1])/ds2_dx2 - diff(det, x[1])/det)
        coeff_0_dx1_ds1 = lambdify(x, coeff_0_dx1_ds1, 'numpy')
        coeff_1_dx1_ds1 = lambdify(x, coeff_1_dx1_ds1, 'numpy')
        coeff_2_dx1_ds1 = lambdify(x, coeff_2_dx1_ds1, 'numpy')
        coeff_0_dx1_ds1 = coeff_0_dx1_ds1(x_value[0], x_value[1])
        coeff_1_dx1_ds1 = coeff_1_dx1_ds1(x_value[0], x_value[1])
        coeff_2_dx1_ds1 = coeff_2_dx1_ds1(x_value[0], x_value[1])
        
        coeff_0_dx1_ds2 = - ds1_dx2/det
        coeff_1_dx1_ds2 = - ds1_dx2/det \
                            *(diff(ds1_dx2, x[0])/ds1_dx2 - diff(det, x[0])/det)
        coeff_2_dx1_ds2 = - ds1_dx2/det \
                            *(diff(ds1_dx2, x[1])/ds1_dx2 - diff(det, x[1])/det)                    
        coeff_0_dx1_ds2 = lambdify(x, coeff_0_dx1_ds2, 'numpy')
        coeff_1_dx1_ds2 = lambdify(x, coeff_1_dx1_ds2, 'numpy')
        coeff_2_dx1_ds2 = lambdify(x, coeff_2_dx1_ds2, 'numpy')
        coeff_0_dx1_ds2 = coeff_0_dx1_ds2(x_value[0], x_value[1])
        coeff_1_dx1_ds2 = coeff_1_dx1_ds2(x_value[0], x_value[1])
        coeff_2_dx1_ds2 = coeff_2_dx1_ds2(x_value[0], x_value[1])
        
        coeff_0_dx2_ds1 = - ds2_dx1/det
        coeff_1_dx2_ds1 = - ds2_dx1/det \
                            *(diff(ds2_dx1, x[0])/ds2_dx1 - diff(det, x[0])/det)
        coeff_2_dx2_ds1 = - ds2_dx1/det \
                            *(diff(ds2_dx1, x[1])/ds2_dx1 - diff(det, x[1])/det)
        coeff_0_dx2_ds1 = lambdify(x, coeff_0_dx2_ds1, 'numpy')
        coeff_1_dx2_ds1 = lambdify(x, coeff_1_dx2_ds1, 'numpy')
        coeff_2_dx2_ds1 = lambdify(x, coeff_2_dx2_ds1, 'numpy')
        coeff_0_dx2_ds1 = coeff_0_dx2_ds1(x_value[0], x_value[1])
        coeff_1_dx2_ds1 = coeff_1_dx2_ds1(x_value[0], x_value[1])
        coeff_2_dx2_ds1 = coeff_2_dx2_ds1(x_value[0], x_value[1])
        
        coeff_0_dx2_ds2 = ds1_dx1/det
        coeff_1_dx2_ds2 = ds1_dx1/det \
                          *(diff(ds1_dx1, x[0])/ds1_dx1 - diff(det, x[0])/det)
        coeff_2_dx2_ds2 = ds1_dx1/det \
                          *(diff(ds1_dx1, x[1])/ds1_dx1 - diff(det, x[1])/det)
        coeff_0_dx2_ds2 = lambdify(x, coeff_0_dx2_ds2, 'numpy')
        coeff_1_dx2_ds2 = lambdify(x, coeff_1_dx2_ds2, 'numpy')
        coeff_2_dx2_ds2 = lambdify(x, coeff_2_dx2_ds2, 'numpy')
        coeff_0_dx2_ds2 = coeff_0_dx2_ds2(x_value[0], x_value[1])
        coeff_1_dx2_ds2 = coeff_1_dx2_ds2(x_value[0], x_value[1])
        coeff_2_dx2_ds2 = coeff_2_dx2_ds2(x_value[0], x_value[1])
        
        modified_dx1_ds1 = coeff_0_dx1_ds1 \
                           + coeff_1_dx1_ds1*x[0] \
                           + coeff_2_dx1_ds1*x[1]
        modified_dx1_ds2 = coeff_0_dx1_ds2 \
                           + coeff_1_dx1_ds2*x[0] \
                           + coeff_2_dx1_ds2*x[1]
        modified_dx2_ds1 = coeff_0_dx2_ds1 \
                           + coeff_1_dx2_ds1*x[0] \
                           + coeff_2_dx2_ds1*x[1]
        modified_dx2_ds2 = coeff_0_dx2_ds2 \
                           + coeff_1_dx2_ds2*x[0] \
                           + coeff_2_dx2_ds2*x[1]
        
        return sym.Matrix([[modified_dx1_ds1, modified_dx1_ds2],
                           [modified_dx2_ds1, modified_dx2_ds2]
                           ])
  
    def modified_x_taylor_dg_ds1(self, x):
        """ 2nd Order Modified x_Taylor Series of dg11/ds1 """
        """ dg11/ds1 = dx1/ds1*dg11/dx1 + dx2/ds1*dg11/dx2 """
        """ dg12/ds1 = dx1/ds1*dg12/dx1 + dx2/ds1*dg12/dx2 """
        """ dg21/ds1 = dx1/ds1*dg21/dx1 + dx2/ds1*dg21/dx2 """
        """ dg22/ds1 = dx1/ds1*dg22/dx1 + dx2/ds1*dg22/dx2 """
        modified_dx1_ds1 = self.modified_x_taylor_dx_ds(x)[0, 0]
        modified_dx2_ds1 = self.modified_x_taylor_dx_ds(x)[1, 0]
        g11 = self.x_taylor_submetric(x)[0, 0]
        g12 = self.x_taylor_submetric(x)[0, 1]
        g21 = self.x_taylor_submetric(x)[1, 0]
        g22 = self.x_taylor_submetric(x)[1, 1]
        
        dg11_dx1 = diff(g11, x[0])
        dg11_dx2 = diff(g11, x[1])
        dg12_dx1 = diff(g12, x[0])
        dg12_dx2 = diff(g12, x[1])
        dg21_dx1 = diff(g21, x[0])
        dg21_dx2 = diff(g21, x[1])
        dg22_dx1 = diff(g22, x[0])
        dg22_dx2 = diff(g22, x[1])
        
        dg11_ds1 = modified_dx1_ds1*dg11_dx1 + modified_dx2_ds1*dg11_dx2
        dg12_ds1 = modified_dx1_ds1*dg12_dx1 + modified_dx2_ds1*dg12_dx2
        dg21_ds1 = modified_dx1_ds1*dg21_dx1 + modified_dx2_ds1*dg21_dx2
        dg22_ds1 = modified_dx1_ds1*dg22_dx1 + modified_dx2_ds1*dg22_dx2
        
        return sym.Matrix([[dg11_ds1, dg12_ds1],
                           [dg21_ds1, dg22_ds1]
                           ])
    
    def term_modified_x_taylor_laplacian_u(self, s, x):
        """ 1st Order x_Taylor Series of Laplacian of u """
        """ 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
        x_value = self.x_values[0][0]
        du_ds1 = self.x_taylor_du_ds(s, x)[0]
        ddu_dds1 = self.x_taylor_ddu_dds(s, x)[0, 0]
        g11 = self.x_taylor_submetric(x)[0, 0]
        g22 = self.x_taylor_submetric(x)[1, 1]
        modified_dg11_ds1 = self.modified_x_taylor_dg_ds1(x)[0, 0]
        modified_dg22_ds1 = self.modified_x_taylor_dg_ds1(x)[1, 1]
        
        laplacian_u = 2*g11*g22*ddu_dds1 \
                      + (g11*modified_dg22_ds1 - \
                         g22*modified_dg11_ds1)*du_ds1
        
        coeff_0_laplacian_u = lambdify(x, laplacian_u, 'numpy')
        coeff_1_laplacian_u = lambdify(x, diff(laplacian_u, x[0]), 'numpy')
        coeff_2_laplacian_u = lambdify(x, diff(laplacian_u, x[1]), 'numpy')
        coeff_0_laplacian_u = coeff_0_laplacian_u(x_value[0], x_value[1])
        coeff_1_laplacian_u = coeff_1_laplacian_u(x_value[0], x_value[1])
        coeff_2_laplacian_u = coeff_2_laplacian_u(x_value[0], x_value[1])

        test = []
        test = [coeff_0_laplacian_u, 
                coeff_1_laplacian_u,
                coeff_2_laplacian_u]
        return test
    
    def term_x_taylor_g12(self, x):
        g12 = self.x_taylor_submetric(x)[0, 1]
        test = []
        for i in range(len(Poly(g12, x).coeffs())):
            temp = Poly(g12, x).coeffs()[i]
            test.append(temp)
        return test
    
    def solution(self, s, x):
#        u_theory = self.u_theory
#        x_value = self.x_values[0][0]
        unknown = self.unknown
        a_theory = self.a_theory[0][0]
        b_theory = self.b_theory[0][0]
        
        f0 = self.term_x_taylor_g12(x)[0]
        f1 = self.term_x_taylor_g12(x)[1]
        f2 = self.term_x_taylor_g12(x)[2]
        f3 = self.term_x_taylor_g12(x)[3]
        f4 = self.term_x_taylor_g12(x)[4]
        f5 = self.term_x_taylor_g12(x)[5]
        f6 = self.term_modified_x_taylor_laplacian_u(s, x)[0]
        unknown_init = ((1 + random.uniform(-0.1, 0.1)/10)*a_theory[3],
                        (1 + random.uniform(-0.1, 0.1)/10)*a_theory[4],
                        (1 + random.uniform(-0.1, 0.1)/10)*a_theory[5],
                        (1 + random.uniform(-0.1, 0.1)/10)*b_theory[3],
                        (1 + random.uniform(-0.1, 0.1)/10)*b_theory[4],
                        (1 + random.uniform(-0.1, 0.1)/10)*b_theory[5]
                        )
        
        coeff_f = np.ndarray((7,), 'object')
        coeff_f[6] = f0
        for i in range(6):
            coeff_f[i] = diff(f0, unknown[i])
        for i in range(7):
            coeff_f[i] = lambdify(unknown, coeff_f[i], 'numpy')
            coeff_f[i] = coeff_f[i](unknown_init[0],
                                    unknown_init[1],
                                    unknown_init[2],
                                    unknown_init[3],
                                    unknown_init[4],
                                    unknown_init[5]
                                    )
        f0 = coeff_f[6] \
             + coeff_f[0]*unknown[0] \
             + coeff_f[1]*unknown[1] \
             + coeff_f[2]*unknown[2] \
             + coeff_f[3]*unknown[3] \
             + coeff_f[4]*unknown[4] \
             + coeff_f[5]*unknown[5]  
             
        coeff_f = np.ndarray((7,), 'object')
        coeff_f[6] = f1
        for i in range(6):
            coeff_f[i] = diff(f1, unknown[i])
        for i in range(7):
            coeff_f[i] = lambdify(unknown, coeff_f[i], 'numpy')
            coeff_f[i] = coeff_f[i](unknown_init[0],
                                    unknown_init[1],
                                    unknown_init[2],
                                    unknown_init[3],
                                    unknown_init[4],
                                    unknown_init[5]
                                    )
        f1 = coeff_f[6] \
             + coeff_f[0]*unknown[0] \
             + coeff_f[1]*unknown[1] \
             + coeff_f[2]*unknown[2] \
             + coeff_f[3]*unknown[3] \
             + coeff_f[4]*unknown[4] \
             + coeff_f[5]*unknown[5]  
        
        coeff_f = np.ndarray((7,), 'object')
        coeff_f[6] = f2
        for i in range(6):
            coeff_f[i] = diff(f2, unknown[i])
        for i in range(7):
            coeff_f[i] = lambdify(unknown, coeff_f[i], 'numpy')
            coeff_f[i] = coeff_f[i](unknown_init[0],
                                    unknown_init[1],
                                    unknown_init[2],
                                    unknown_init[3],
                                    unknown_init[4],
                                    unknown_init[5]
                                    )
        f2 = coeff_f[6] \
             + coeff_f[0]*unknown[0] \
             + coeff_f[1]*unknown[1] \
             + coeff_f[2]*unknown[2] \
             + coeff_f[3]*unknown[3] \
             + coeff_f[4]*unknown[4] \
             + coeff_f[5]*unknown[5]  
             
        coeff_f = np.ndarray((7,), 'object')
        coeff_f[6] = f3
        for i in range(6):
            coeff_f[i] = diff(f3, unknown[i])
        for i in range(7):
            coeff_f[i] = lambdify(unknown, coeff_f[i], 'numpy')
            coeff_f[i] = coeff_f[i](unknown_init[0],
                                    unknown_init[1],
                                    unknown_init[2],
                                    unknown_init[3],
                                    unknown_init[4],
                                    unknown_init[5]
                                    )
        f3 = coeff_f[6] \
             + coeff_f[0]*unknown[0] \
             + coeff_f[1]*unknown[1] \
             + coeff_f[2]*unknown[2] \
             + coeff_f[3]*unknown[3] \
             + coeff_f[4]*unknown[4] \
             + coeff_f[5]*unknown[5]  
             
        coeff_f = np.ndarray((7,), 'object')
        coeff_f[6] = f4
        for i in range(6):
            coeff_f[i] = diff(f4, unknown[i])
        for i in range(7):
            coeff_f[i] = lambdify(unknown, coeff_f[i], 'numpy')
            coeff_f[i] = coeff_f[i](unknown_init[0],
                                    unknown_init[1],
                                    unknown_init[2],
                                    unknown_init[3],
                                    unknown_init[4],
                                    unknown_init[5]
                                    )
        f4 = coeff_f[6] \
             + coeff_f[0]*unknown[0] \
             + coeff_f[1]*unknown[1] \
             + coeff_f[2]*unknown[2] \
             + coeff_f[3]*unknown[3] \
             + coeff_f[4]*unknown[4] \
             + coeff_f[5]*unknown[5]  
        
        coeff_f = np.ndarray((7,), 'object')
        coeff_f[6] = f5
        for i in range(6):
            coeff_f[i] = diff(f5, unknown[i])
        for i in range(7):
            coeff_f[i] = lambdify(unknown, coeff_f[i], 'numpy')
            coeff_f[i] = coeff_f[i](unknown_init[0],
                                    unknown_init[1],
                                    unknown_init[2],
                                    unknown_init[3],
                                    unknown_init[4],
                                    unknown_init[5]
                                    )
        f5 = coeff_f[6] \
             + coeff_f[0]*unknown[0] \
             + coeff_f[1]*unknown[1] \
             + coeff_f[2]*unknown[2] \
             + coeff_f[3]*unknown[3] \
             + coeff_f[4]*unknown[4] \
             + coeff_f[5]*unknown[5]  
             
        coeff_f = np.ndarray((7,), 'object')
        coeff_f[6] = f6
        for i in range(6):
            coeff_f[i] = diff(f6, unknown[i])
        for i in range(7):
            coeff_f[i] = lambdify(unknown, coeff_f[i], 'numpy')
            coeff_f[i] = coeff_f[i](unknown_init[0],
                                    unknown_init[1],
                                    unknown_init[2],
                                    unknown_init[3],
                                    unknown_init[4],
                                    unknown_init[5]
                                    )
        f6 = coeff_f[6] \
             + coeff_f[0]*unknown[0] \
             + coeff_f[1]*unknown[1] \
             + coeff_f[2]*unknown[2] \
             + coeff_f[3]*unknown[3] \
             + coeff_f[4]*unknown[4] \
             + coeff_f[5]*unknown[5]  

        f = (f0, f1, f2, f3, f4, f5, f6)
        solution = nsolve(f, unknown, unknown_init)
        
#        x_taylor_u = self.x_taylor_u(known, unknown, s, x)
#        u_experiment = lambdify([unknown, x], x_taylor_u, 'numpy')
#        u_experiment = u_experiment((a_experiment[0],
#                                     a_experiment[1],
#                                     a_experiment[2],
#                                     b_experiment[0],
#                                     b_experiment[1],
#                                     b_experiment[2]),
#                                    (x_value[0],
#                                     x_value[1])
#                                    )
#                                    
#        u_theory = lambdify(x, u_theory, 'numpy')
#        u_theory = u_theory(x_value[0], x_value[1])     
#        
#        result = np.ndarray((2,))
#        result[0] = u_theory
#        result[1] = u_experiment
        return solution

        

if __name__ == '__main__':
    
    t0 = time.time()
    
    s = [Symbol('s1', real = True), 
         Symbol('s2', real = True)
         ]
    x = [Symbol('x1', real = True), 
         Symbol('x2', real = True)
         ]
    unknown = [Symbol('a11', real = True),
               Symbol('a12', real = True),
               Symbol('a22', real = True),
               Symbol('b11', real = True),
               Symbol('b12', real = True),
               Symbol('b22', real = True)
               ]
 
    theory = laplace_theory.Theory()
    r_theory = theory.r_theory(x)[0][0]
    a_theory = theory.a_theory(x)[0][0]
    b_theory = theory.b_theory(x)[0][0]
    
    print('x_values = ')
    print(theory.x_values(x))
    print('')
    
    temp = [a_theory[3], 
            a_theory[4],
            a_theory[5],
            b_theory[3],
            b_theory[4],
            b_theory[5]] 
    print('(a_theory, b_theory) = ')
    [print(round(item, 4)) for item in temp]
    print('')
    
    
    experiment = Experiment(s, x)
    temp = experiment.solution(s, x)
    
    print('(a_experiment, b_experiment) = ')
    [print(round(item, 4)) for item in temp]
    print('')
    
    
    # Verification
    g12 = experiment.term_x_taylor_g12(x)
    g12 = lambdify(unknown, g12, 'numpy')
    g12 = g12(a_theory[3],
              a_theory[4],
              a_theory[5],
              b_theory[3],
              b_theory[4],
              b_theory[5])
    print('Verificaiton of g12')
    [print(round(item, 4)) for item in g12]
    print('')
    
    laplacian_u = experiment.term_modified_x_taylor_laplacian_u(s, x)
    laplacian_u = lambdify(unknown, laplacian_u, 'numpy')
    laplacian_u = laplacian_u(a_theory[3],
                              a_theory[4],
                              a_theory[5],
                              b_theory[3],
                              b_theory[4],
                              b_theory[5])
    print('Verificaiton of Laplacian u')
    [print(round(item, 4)) for item in laplacian_u]
    print('')

    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
        
    










