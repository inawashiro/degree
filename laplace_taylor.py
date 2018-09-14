# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:00:27 2018

@author: inawashiro
"""
import laplace_theory


# For Visualization
import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle']='solid'

# For Numerical Computation 
import numpy as np

# For Symbolic Notation
import sympy as sym
from sympy import Symbol, diff, Matrix, simplify, factor, S, Poly, nsolve, lambdify
sym.init_printing()

# For Random Variables
import random

# For Symbolic Mathematics
from IPython.display import display

# For measuring computation time
import time



class Taylor_Functions(laplace_theory.Theoretical_Values):
    """ Tayolr Series Expression of Parameters """
    
    def __init__(self, known, unknown, s, x):
        self.theory = laplace_theory.Theoretical_Values()
        self.x_values = self.theory.x_values(x)
        self.r_theory = self.theory.r_theory(x)
        self.a_theory = self.theory.a_theory(x)
        self.b_theory = self.theory.b_theory(x)

    def s_taylor_u(self, known, unknown, s):
        """ 2nd Order s_Taylor Series of u """
        return known[0] \
               + unknown[0]*s[0] \
               + unknown[1]*s[1] \
               + unknown[2]*s[0]**2 \
               + unknown[3]*s[0]*s[1] \
               + unknown[4]*s[1]**2
    
    def x_taylor_s1(self, known, unknown, x):
        """ 2nd Order x_Taylor Series of s1 """
        return known[1] \
               + known[2]*x[0] \
               + known[3]*x[1] \
               + unknown[5]*x[0]**2 \
               + unknown[6]*x[0]*x[1] \
               + unknown[7]*x[1]**2
        
    def x_taylor_s2(self, known, unknown, x):
        """ 2nd Order x_Taylor Series of s2 """
        return known[4] \
               + known[5]*x[0] \
               + known[6]*x[1] \
               + unknown[8]*x[0]**2 \
               + unknown[9]*x[0]*x[1] \
               + unknown[10]*x[1]**2    
    
    def s_taylor_du_ds(self, known, unknown, s):
        """ 1st Order s_Taylor Series of (du/ds) """
        u = self.s_taylor_u(known, unknown, s)
        du_ds1 = diff(u, s[0])
        du_ds2 = diff(u, s[1])
        return sym.Matrix([du_ds1,
                           du_ds2])
                
    def x_taylor_du_ds(self, known, unknown, s, x):
        """ 1st Order x_Taylor Series of (du/ds) """
        du_ds1 = self.s_taylor_du_ds(known, unknown, s)[0]
        du_ds2 = self.s_taylor_du_ds(known, unknown, s)[1]
        s1 = self.x_taylor_s1(known, unknown, x)
        s2 = self.x_taylor_s2(known, unknown, x)
        return sym.Matrix([du_ds1.subs([(s[0], s1), (s[1], s2)]),
                           du_ds2.subs([(s[0], s1), (s[1], s2)])
                           ])
        
    def x_taylor_ddu_dds(self, known, unknown, s, x):
        """ 0th Order x_Taylor Series of (ddu/dds) """
        u = self.s_taylor_u(known, unknown, s)
        ddu_dds1 = diff(u, s[0], 2)
        ddu_ds1ds2 = diff(u, s[0], s[1])
        ddu_dds2 = diff(u, s[1], 2)
        s1 = self.x_taylor_s1(known, unknown, x)
        s2 = self.x_taylor_s2(known, unknown, x)
        return sym.Matrix([[ddu_dds1.subs([(s[0], s1), (s[1], s2)]), 
                            ddu_ds1ds2.subs([(s[0], s1), (s[1], s2)])],
                           [ddu_ds1ds2.subs([(s[0], s1), (s[1], s2)]),
                            ddu_dds2.subs([(s[0], s1), (s[1], s2)])]
                           ])
       
    def x_taylor_ds_dx(self, known, unknown, x):
        """ 1st Order x_Taylor Series of (ds/dx) """
        s1 = self.x_taylor_s1(known, unknown, x)
        s2 = self.x_taylor_s2(known, unknown, x)
        ds1_dx1 = diff(s1, x[0])
        ds1_dx2 = diff(s1, x[1])
        ds2_dx1 = diff(s2, x[0])
        ds2_dx2 = diff(s2, x[1])
        return sym.Matrix([[ds1_dx1, ds1_dx2],
                           [ds2_dx1, ds2_dx2]
                           ])
        
    def x_taylor_submetric(self, known, unknown, x):
        """ 2nd Order x_Taylor Series of Subscript Metric g_ij """
        ds_dx = self.x_taylor_ds_dx(known, unknown, x)
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
        
    def modified_x_taylor_dx_ds(self, known, unknown, x):
        """ 1st Order Modified x_Taylor Series of (dx/ds) """
        x_value = self.x_values[0][0]
        ds_dx = self.x_taylor_ds_dx(known, unknown, x)
        
#        ds1_dx1 = ds_dx[0, 0]
#        ds1_dx2 = ds_dx[0, 1]
#        ds2_dx1 = ds_dx[1, 0]
#        ds2_dx2 = ds_dx[1, 1]
#               
#        det = ds1_dx1*ds2_dx2 - ds1_dx2*ds2_dx1
#        
#        coeff_0_det = lambdify(x, det, 'numpy')
#        coeff_1_det = lambdify(x, diff(det, x[0]), 'numpy')
#        coeff_2_det = lambdify(x, diff(det, x[1]), 'numpy')
#        coeff_0_det = coeff_0_det(x_value[0], x_value[1])
#        coeff_1_det = coeff_1_det(x_value[0], x_value[1])
#        coeff_2_det = coeff_2_det(x_value[0], x_value[1])
#        
#        coeff_0_ds1_dx1 = lambdify(x, ds1_dx1, 'numpy')
#        coeff_1_ds1_dx1 = lambdify(x, diff(ds1_dx1, x[0]), 'numpy')
#        coeff_2_ds1_dx1 = lambdify(x, diff(ds1_dx1, x[1]), 'numpy')
#        coeff_0_ds1_dx1 = coeff_0_ds1_dx1(x_value[0], x_value[1])
#        coeff_1_ds1_dx1 = coeff_1_ds1_dx1(x_value[0], x_value[1])
#        coeff_2_ds1_dx1 = coeff_2_ds1_dx1(x_value[0], x_value[1])
#        
#        coeff_0_ds1_dx2 = lambdify(x, ds1_dx2, 'numpy')
#        coeff_1_ds1_dx2 = lambdify(x, diff(ds1_dx2, x[0]), 'numpy')
#        coeff_2_ds1_dx2 = lambdify(x, diff(ds1_dx2, x[1]), 'numpy')
#        coeff_0_ds1_dx2 = coeff_0_ds1_dx2(x_value[0], x_value[1])
#        coeff_1_ds1_dx2 = coeff_1_ds1_dx2(x_value[0], x_value[1])
#        coeff_2_ds1_dx2 = coeff_2_ds1_dx2(x_value[0], x_value[1])
#        
#        coeff_0_ds2_dx1 = lambdify(x, ds2_dx1, 'numpy')
#        coeff_1_ds2_dx1 = lambdify(x, diff(ds2_dx1, x[0]), 'numpy')
#        coeff_2_ds2_dx1 = lambdify(x, diff(ds2_dx1, x[1]), 'numpy')
#        coeff_0_ds2_dx1 = coeff_0_ds2_dx1(x_value[0], x_value[1])
#        coeff_1_ds2_dx1 = coeff_1_ds2_dx1(x_value[0], x_value[1])
#        coeff_2_ds2_dx1 = coeff_2_ds2_dx1(x_value[0], x_value[1])
#        
#        coeff_0_ds2_dx2 = lambdify(x, ds2_dx2, 'numpy')
#        coeff_1_ds2_dx2 = lambdify(x, diff(ds2_dx2, x[0]), 'numpy')
#        coeff_2_ds2_dx2 = lambdify(x, diff(ds2_dx2, x[1]), 'numpy')
#        coeff_0_ds2_dx2 = coeff_0_ds2_dx2(x_value[0], x_value[1])
#        coeff_1_ds2_dx2 = coeff_1_ds2_dx2(x_value[0], x_value[1])
#        coeff_2_ds2_dx2 = coeff_2_ds2_dx2(x_value[0], x_value[1])
#        
#        coeff_0_dx1_ds1 = coeff_0_ds2_dx2/coeff_0_det
#        coeff_1_dx1_ds1 = coeff_0_ds2_dx2/coeff_0_det \
#                          *(coeff_1_ds2_dx2/coeff_0_ds2_dx2 \
#                            - coeff_1_det/coeff_0_det)       
#        coeff_2_dx1_ds1 = coeff_0_ds2_dx2/coeff_0_det \
#                          *(coeff_2_ds2_dx2/coeff_0_ds2_dx2 \
#                            - coeff_2_det/coeff_0_det) 
#                          
#        coeff_0_dx1_ds2 = - coeff_0_ds1_dx2/coeff_0_det
#        coeff_1_dx1_ds2 = - coeff_0_ds1_dx2/coeff_0_det \
#                          *(coeff_1_ds1_dx2/coeff_0_ds1_dx2 \
#                            - coeff_1_det/coeff_0_det)       
#        coeff_2_dx1_ds2 = coeff_0_ds1_dx2/coeff_0_det \
#                          *(coeff_2_ds1_dx2/coeff_0_ds1_dx2 \
#                            - coeff_2_det/coeff_0_det)
#                          
#        coeff_0_dx2_ds1 = - coeff_0_ds2_dx2/coeff_0_det
#        coeff_1_dx2_ds1 = - coeff_0_ds2_dx1/coeff_0_det \
#                          *(coeff_1_ds2_dx1/coeff_0_ds2_dx1 \
#                            - coeff_1_det/coeff_0_det)       
#        coeff_2_dx2_ds1 = coeff_0_ds2_dx1/coeff_0_det \
#                          *(coeff_2_ds2_dx1/coeff_0_ds1_dx2 \
#                            - coeff_2_det/coeff_0_det)
#                          
#        coeff_0_dx2_ds2 = coeff_0_ds1_dx1/coeff_0_det
#        coeff_1_dx2_ds2 = coeff_0_ds1_dx1/coeff_0_det \
#                          *(coeff_1_ds1_dx1/coeff_0_ds1_dx1 \
#                            - coeff_1_det/coeff_0_det)       
#        coeff_2_dx2_ds2 = coeff_0_ds1_dx1/coeff_0_det \
#                          *(coeff_2_ds1_dx1/coeff_0_ds1_dx1 \
#                            - coeff_2_det/coeff_0_det)
#        
#        modified_dx1_ds1 = coeff_0_dx1_ds1 \
#                           + coeff_1_dx1_ds1*x[0] \
#                           + coeff_2_dx1_ds1*x[1]
#        modified_dx1_ds2 = coeff_0_dx1_ds2 \
#                           + coeff_1_dx1_ds2*x[0] \
#                           + coeff_2_dx1_ds2*x[1]
#        modified_dx2_ds1 = coeff_0_dx2_ds1 \
#                           + coeff_1_dx2_ds1*x[0] \
#                           + coeff_2_dx2_ds1*x[1]
#        modified_dx2_ds2 = coeff_0_dx2_ds2 \
#                           + coeff_1_dx2_ds2*x[0] \
#                           + coeff_2_dx2_ds2*x[1]
#        
#        return sym.Matrix([[modified_dx1_ds1, modified_dx1_ds2],
#                           [modified_dx2_ds1, modified_dx2_ds2]
#                           ])
            
            
        dx1_ds1 = ds_dx.inv()[0, 0]
        dx1_ds2 = ds_dx.inv()[0, 1]
        dx2_ds1 = ds_dx.inv()[1, 0]
        dx2_ds2 = ds_dx.inv()[1, 1]
        x_value = self.x_values[0][0]
        
        coeff_0_dx1_ds1 = lambdify(x, dx1_ds1, 'numpy')
        coeff_0_dx1_ds2 = lambdify(x, dx1_ds2, 'numpy')
        coeff_0_dx2_ds1 = lambdify(x, dx2_ds1, 'numpy')
        coeff_0_dx2_ds2 = lambdify(x, dx2_ds2, 'numpy')
        
        coeff_1_dx1_ds1 = lambdify(x, diff(dx1_ds1, x[0]), 'numpy')
        coeff_1_dx1_ds2 = lambdify(x, diff(dx1_ds2, x[0]), 'numpy')
        coeff_1_dx2_ds1 = lambdify(x, diff(dx2_ds1, x[0]), 'numpy')
        coeff_1_dx2_ds2 = lambdify(x, diff(dx2_ds2, x[0]), 'numpy')
        
        coeff_2_dx1_ds1 = lambdify(x, diff(dx1_ds1, x[1]), 'numpy')
        coeff_2_dx1_ds2 = lambdify(x, diff(dx1_ds2, x[1]), 'numpy')
        coeff_2_dx2_ds1 = lambdify(x, diff(dx2_ds1, x[1]), 'numpy')
        coeff_2_dx2_ds2 = lambdify(x, diff(dx2_ds2, x[1]), 'numpy')
        
        coeff_0_dx1_ds1 = coeff_0_dx1_ds1(x_value[0], x_value[1])
        coeff_0_dx1_ds2 = coeff_0_dx1_ds2(x_value[0], x_value[1])
        coeff_0_dx2_ds1 = coeff_0_dx2_ds1(x_value[0], x_value[1])
        coeff_0_dx2_ds2 = coeff_0_dx2_ds2(x_value[0], x_value[1])
        
        coeff_1_dx1_ds1 = coeff_1_dx1_ds1(x_value[0], x_value[1])
        coeff_1_dx1_ds2 = coeff_1_dx1_ds2(x_value[0], x_value[1])
        coeff_1_dx2_ds1 = coeff_1_dx2_ds1(x_value[0], x_value[1])
        coeff_1_dx2_ds2 = coeff_1_dx2_ds2(x_value[0], x_value[1])
        
        coeff_2_dx1_ds1 = coeff_2_dx1_ds1(x_value[0], x_value[1])
        coeff_2_dx1_ds2 = coeff_2_dx1_ds2(x_value[0], x_value[1])
        coeff_2_dx2_ds1 = coeff_2_dx2_ds1(x_value[0], x_value[1])
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
  
    def modified_x_taylor_dg11_ds1(self, known, unknown, x):
        """ 2nd Order Modified x_Taylor Series of dg11/ds1 """
        """ dg11/ds1 = dx1/ds1*dg11/dx1 + dx2/ds1*dg11/dx2 """
#        x_value = self.x_values[0][0]
        modified_dx1_ds1 = self.modified_x_taylor_dx_ds(known, unknown, x)[0, 0]
        modified_dx2_ds1 = self.modified_x_taylor_dx_ds(known, unknown, x)[1, 0]
        g11 = self.x_taylor_submetric(known, unknown, x)[0, 0]
        
        dg11_dx1 = diff(g11, x[0])
        dg11_dx2 = diff(g11, x[1])
        temp = modified_dx1_ds1*dg11_dx1 + modified_dx2_ds1*dg11_dx2
        
        return temp
    
    def modified_x_taylor_dg22_ds1(self, known, unknown, x):
        """ 2nd Order x_Taylor Series of dg22/ds1 """
        """ dg22/ds1 = dx1/ds1*dg22/dx1 + dx2/ds1*dg22/dx2 """
#        x_value = self.x_values[0][0]
        modified_dx1_ds1 = self.modified_x_taylor_dx_ds(known, unknown, x)[0, 0]
        modified_dx2_ds1 = self.modified_x_taylor_dx_ds(known, unknown, x)[1, 0]
        
        g22 = self.x_taylor_submetric(known, unknown, x)[1, 1]
        dg22_dx1 = diff(g22, x[0])
        dg22_dx2 = diff(g22, x[1])
        temp = modified_dx1_ds1*dg22_dx1 + modified_dx2_ds1*dg22_dx2
        
        return temp
    
    def modified_x_taylor_laplacian_u(self, known, unknown, s, x):
        """ 1st Order x_Taylor Series of Laplacian of u """
        """ 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
        x_value = self.x_values[0][0]
        # u,1
        # 2nd Order
        du_ds1 = self.x_taylor_du_ds(known, unknown, s, x)[0]
        # u,11
        # 0th Order
        ddu_dds1 = self.x_taylor_ddu_dds(known, unknown, s, x)[0, 0]
        # g11
        # 2nd Order
        g11 = self.x_taylor_submetric(known, unknown, x)[0, 0]
        # g22
        # 2nd Order
        g22 = self.x_taylor_submetric(known, unknown, x)[1, 1]
        # g11,1
        # 2nd Order
        modified_dg11_ds1 = self.modified_x_taylor_dg11_ds1(known, unknown, x)
        # g22,1
        # 2nd Order
        modified_dg22_ds1 = self.modified_x_taylor_dg22_ds1(known, unknown, x)
        # 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1
        # 3rd Order
        temp = 2*g11*g22*ddu_dds1 \
               + (g11*modified_dg22_ds1 - \
                  modified_dg11_ds1*g22)*du_ds1
        
        coeff_0_laplacian_u = lambdify(x, temp, 'numpy')
        coeff_1_laplacian_u = lambdify(x, diff(temp, x[0]), 'numpy')
        coeff_2_laplacian_u = lambdify(x, diff(temp, x[1]), 'numpy')
        
        coeff_0_laplacian_u = coeff_0_laplacian_u(x_value[0], x_value[1])
        coeff_1_laplacian_u = coeff_1_laplacian_u(x_value[0], x_value[1])
        coeff_2_laplacian_u = coeff_2_laplacian_u(x_value[0], x_value[1])
        
        test = []
        test = [coeff_0_laplacian_u,
                coeff_1_laplacian_u,
                coeff_2_laplacian_u]
        return test
                
#    def constraint_1_term(self, known, unknown, s):
#        """ Terms of s_Taylor Series of du/ds2 """
#        temp = []
#        constraint_1 = self.s_taylor_du_ds(known, unknown, s)[1]
#        r_theory = self.r_theory[0][0]
#        a_theory = self.a_theory[0][0]
#        b_theory = self.b_theory[0][0]
#        for i in range(len(Poly(constraint_1, s).coeffs())):
#            test = Poly(constraint_1, s).coeffs()[i]
#            test = lambdify(known, test, 'numpy')
#            test = test(r_theory[0], 
#                        a_theory[0],
#                        a_theory[1],
#                        a_theory[2],
#                        b_theory[0],
#                        b_theory[1],
#                        b_theory[2]
#                        )
#            temp.append(test)
#        return temp
#    
#    def constraint_1_term_verification(self, known, unknown, s):
#        temp = []
#        constraint_1_term = self.constraint_1_term
#        r_theory = self.r_theory[0][0]
#        a_theory = self.a_theory[0][0]
#        b_theory = self.b_theory[0][0]
#        for i in range(len(constraint_1_term(known, unknown, s))):
#            test = constraint_1_term(known, unknown, s)[i]
#            test = lambdify(unknown, test, 'numpy')
#            test = test(r_theory[1], 
#                        r_theory[2], 
#                        r_theory[3],
#                        r_theory[4],
#                        r_theory[5],
#                        a_theory[3],
#                        a_theory[4],
#                        a_theory[5],
#                        b_theory[3],
#                        b_theory[4],
#                        b_theory[5])
#            temp.append(test)
#        return temp
#    
#    def constraint_2_term(self, known, unknown, x):
#        """ Terms of s_Taylor Series of g12 """
#        temp = []
#        constraint_2 = self.x_taylor_submetric(known, unknown, x)[0, 1]
#        r_theory = self.r_theory[0][0]
#        a_theory = self.a_theory[0][0]
#        b_theory = self.b_theory[0][0]
#        for i in range(len(Poly(constraint_2, x).coeffs())):
#            test = Poly(constraint_2, x).coeffs()[i]
#            test = lambdify(known, test, 'numpy')
#            test = test(r_theory[0], 
#                        a_theory[0],
#                        a_theory[1],
#                        a_theory[2],
#                        b_theory[0],
#                        b_theory[1],
#                        b_theory[2]
#                        )
#            temp.append(test)
#        return temp
#    
#    def constraint_2_term_verification(self, known, unknown, x):
#        temp = []
#        constraint_2_term = self.constraint_2_term
#        r_theory = self.r_theory[0][0]
#        a_theory = self.a_theory[0][0]
#        b_theory = self.b_theory[0][0]
#        for i in range(len(constraint_2_term(known, unknown, x))):
#            test = constraint_2_term(known, unknown, x)[i]
#            test = lambdify(unknown, test, 'numpy')
#            test = test(r_theory[1], 
#                        r_theory[2], 
#                        r_theory[3],
#                        r_theory[4],
#                        r_theory[5],
#                        a_theory[3],
#                        a_theory[4],
#                        a_theory[5],
#                        b_theory[3],
#                        b_theory[4],
#                        b_theory[5])
#            temp.append(test)
#        return temp
#    
#    def constraint_3_term(self, known, unknown, s, x):
#        """ Terms of s_Taylor Series of 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
#        temp = []
#        constraint_3 = self.modified_x_taylor_laplacian_u(known, unknown, s, x)
#        r_theory = self.r_theory[0][0]
#        a_theory = self.a_theory[0][0]
#        b_theory = self.b_theory[0][0]
#        for i in range(len(Poly(constraint_3, x).coeffs())):
#            test = Poly(constraint_3, x).coeffs()[i]
#            test = lambdify(known, test, 'numpy')
#            test = test(r_theory[0], 
#                        a_theory[0],
#                        a_theory[1],
#                        a_theory[2],
#                        b_theory[0],
#                        b_theory[1],
#                        b_theory[2]
#                        )
#            temp.append(test)
#        return temp
#    
#    def constraint_3_term_verification(self, known, unknown, s, x):
#        temp = []
#        constraint_3_term = self.constraint_3_term
#        r_theory = self.r_theory[0][0]
#        a_theory = self.a_theory[0][0]
#        b_theory = self.b_theory[0][0]
#        for i in range(len(constraint_3_term(known, unknown, s, x))):
#            test = constraint_3_term(known, unknown, x)[i]
#            test = lambdify(unknown, test, 'numpy')
#            test = test(r_theory[1], 
#                        r_theory[2], 
#                        r_theory[3],
#                        r_theory[4],
#                        r_theory[5],
#                        a_theory[3],
#                        a_theory[4],
#                        a_theory[5],
#                        b_theory[3],
#                        b_theory[4],
#                        b_theory[5])
#            temp.append(test)
#        return temp

    def solution(self, known, unknown, s, x):
        r_theory = self.r_theory[0][0]
        a_theory = self.a_theory[0][0]
        b_theory = self.b_theory[0][0]
        
        f1 = self.constraint_1_term(known, unknown, s)[0]
        f2 = self.constraint_1_term(known, unknown, s)[1]
        f3 = self.constraint_1_term(known, unknown, s)[2]
        f4 = self.constraint_2_term(known, unknown, x)[0]
        f5 = self.constraint_2_term(known, unknown, x)[1]
        f6 = self.constraint_2_term(known, unknown, x)[2]
        f7 = self.constraint_2_term(known, unknown, x)[3]
        f8 = self.constraint_2_term(known, unknown, x)[4]
        f9 = self.constraint_3_term(known, unknown, s, x)[0]
        f10 = self.constraint_3_term(known, unknown, s, x)[1]
        f11 = self.constraint_3_term(known, unknown, s, x)[2]
        f = (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11)
        
        unknown_init = ((1 + random.uniform(-1.0, 1.0)/10)*r_theory[1],
                        (1 + random.uniform(-1.0, 1.0)/10)*r_theory[2],
                        (1 + random.uniform(-1.0, 1.0)/10)*r_theory[3],
                        (1 + random.uniform(-1.0, 1.0)/10)*r_theory[4],
                        (1 + random.uniform(-1.0, 1.0)/10)*r_theory[5],
                        (1 + random.uniform(-1.0, 1.0)/10)*a_theory[3],
                        (1 + random.uniform(-1.0, 1.0)/10)*a_theory[4],
                        (1 + random.uniform(-1.0, 1.0)/10)*a_theory[5],
                        (1 + random.uniform(-1.0, 1.0)/10)*b_theory[3],
                        (1 + random.uniform(-1.0, 1.0)/10)*b_theory[4],
                        (1 + random.uniform(-1.0, 1.0)/10)*b_theory[5]
                        )
        return nsolve(f, unknown, unknown_init)
#    
#    def r_experiment(self, known, unknown, s, x):
#        return [self.solution(known, unknown, s, x)[0],
#                self.solution(known, unknown, s, x)[1],
#                self.solution(known, unknown, s, x)[2],
#                self.solution(known, unknown, s, x)[3],
#                self.solution(known, unknown, s, x)[4]
#                ]
#        
#    def a_experiment(self, known, unknown, s, x):
#        return [self.solution(known, unknown, s, x)[5],
#                self.solution(known, unknown, s, x)[6],
#                self.solution(known, unknown, s, x)[7],
#                ]
#            
#    def b_experiment(self, known, unknown, s, x):
#        return [self.solution(known, unknown, s, x)[8],
#                self.solution(known, unknown, s, x)[9],
#                self.solution(known, unknown, s, x)[10],
#                ]
#        
#    def r_error(self, known, unknown, s, x):
#        return self.r_experiment(known, unknown, s, x) - self.r_theory
#    
#    def a_error(self, known, unknown, s, x):
#        return self.a_experiment(known, unknown, s, x) - self.a_theory
#    
#    def b_error(self, known, unknown, s, x):
#        return self.b_experiment(known, unknown, s, x) - self.b_theory
    
    

        

if __name__ == '__main__':
    
    t0 = time.time()
    
    theory = laplace_theory.Theoretical_Values()
    
    s = [Symbol('s1', real = True), 
         Symbol('s2', real = True)
         ]
    
    x = [Symbol('x1', real = True), 
         Symbol('x2', real = True)
         ]
    
#    known = [Symbol('r0', real = True),
#             Symbol('a0', real = True),
#             Symbol('a1', real = True),
#             Symbol('a2', real = True),
#             Symbol('b0', real = True),
#             Symbol('b1', real = True),
#             Symbol('b2', real = True)
#             ]
    
    known = [theory.r_theory(x)[0][0][0],
             theory.a_theory(x)[0][0][0],
             theory.a_theory(x)[0][0][1],
             theory.a_theory(x)[0][0][2],
             theory.b_theory(x)[0][0][0],
             theory.b_theory(x)[0][0][1],
             theory.b_theory(x)[0][0][2]
             ]
    
    unknown = [Symbol('r1', real = True),
               Symbol('r2', real = True),
               Symbol('r11', real = True),
               Symbol('r12', real = True),
               Symbol('r22', real = True),
               Symbol('a11', real = True),
               Symbol('a12', real = True),
               Symbol('a22', real = True),
               Symbol('b11', real = True),
               Symbol('b12', real = True),
               Symbol('b22', real = True)
               ]
    
    
    print('x_values = ', theory.x_values(x))
    print('')
    
    print('r_theory = ', theory.r_theory(x))
    print('')
    
    print('a_theory = ', theory.a_theory(x))
    print('')
    
    print('b_theory = ', theory.b_theory(x))
    print('')
    
    
    
    taylor = Taylor_Functions(known, unknown, s, x)
    
    print('Modified x_Taylor Series of (dx/ds) = ')
    display(taylor.modified_x_taylor_dx_ds(known, unknown, x))
    print('')
    
    print('Modified x_Taylor Series of dg11/ds1 = ')
    display(taylor.modified_x_taylor_dg11_ds1(known, unknown, x))
    print('')
    
    print('Modified x_Taylor Series of dg22/ds1 = ')
    display(taylor.modified_x_taylor_dg22_ds1(known, unknown, x))
    print('')
    
    print('Modified x_Taylor Series of Laplacian of u = ')
    for i in range(len(taylor.modified_x_taylor_laplacian_u(known, unknown, s, x))):
        display(taylor.modified_x_taylor_laplacian_u(known, unknown, s, x)[i])    
    print('')

    
#    print('s_Taylor Series of du/ds2  = ')
#    print('# of terms =', len(taylor.constraint_1_term(known, unknown, s)))
#    for i in range(len(taylor.constraint_1_term(known, unknown, s))):
#        display(taylor.constraint_1_term(known, unknown, s)[i])
#    print('')
#    print('# of terms =', len(taylor.constraint_1_term_verification(known, unknown, s)))
#    for i in range(len(taylor.constraint_1_term_verification(known, unknown, s))):
#        display(round(taylor.constraint_1_term_verification(known, unknown, s)[i], 4))
#    print('')
#    
#    print('x_Taylor Series of g12  = ')
#    print('# of terms =', len(taylor.constraint_2_term(known, unknown, x)))
#    for i in range(len(taylor.constraint_2_term(known, unknown, x))):
#        display(taylor.constraint_2_term(known, unknown, x)[i])
#    print('')
#    print('# of terms =', len(taylor.constraint_2_term_verification(known, unknown, x)))
#    for i in range(len(taylor.constraint_2_term_verification(known, unknown, x))):
#        display(round(taylor.constraint_2_term_verification(known, unknown, x)[i], 4))
#    print('')
#    
#    print('x_Taylor Series of Laplacian of u  = ')
#    print('# of terms =', len(taylor.constraint_3_term(known, unknown, s, x)))
#    for i in range(len(taylor.constraint_3_term(known, unknown, s, x))):
#        display(taylor.constraint_3_term(known, unknown, s, x)[i])
#    print('')
#    print('# of terms =', len(taylor.constraint_3_term_verification(known, unknown, s, x)))
#    for i in range(len(taylor.constraint_3_term_verification(known, unknown, s, x))):
#        display(round(taylor.constraint_3_term_verification(known, unknown, s, x)[i], 4))
#    print('')
#    
#    
#    
#    print('Solution = ')
#    display(taylor.solution(known, unknown, s, x))
#    print('')
    
#    print('Error of r = ')
#    print(solve.r_error(known, unknown, s, x))
#    print('')
#    
#    print('Error of a = ')
#    print(solve.a_error(known, unknown, s, x))
#    print('')
#    
#    print('Error of b = ')
#    print(solve.b_error(known, unknown, s, x))
#    print('')
    
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
        
    







































