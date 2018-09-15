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

## For Symbolic Mathematics
#from IPython.display import display

# For measuring computation time
import time



class Taylor_Functions(laplace_theory.Theoretical_Values):
    """ Tayolr Series Expression of Parameters """
    
    def __init__(self, known, unknown, s, x):
        self.theory = laplace_theory.Theoretical_Values()
        self.u_theory = self.theory.u(x)
        self.x_values = self.theory.x_values(x)
        self.r_theory = self.theory.r_theory(x)
        self.a_theory = self.theory.a_theory(x)
        self.b_theory = self.theory.b_theory(x)

    def s_taylor_u(self, known, unknown, s):
        """ 2nd Order s_Taylor Series of u """
        return known[0] \
               + known[1]*s[0] \
               + known[2]*s[1] \
               + known[3]*s[0]**2 \
               + known[4]*s[0]*s[1] \
               + known[5]*s[1]**2
    
    def x_taylor_s1(self, known, unknown, x):
        """ 2nd Order x_Taylor Series of s1 """
        return known[6] \
               + known[7]*x[0] \
               + known[8]*x[1] \
               + unknown[0]*x[0]**2 \
               + unknown[1]*x[0]*x[1] \
               + unknown[2]*x[1]**2
        
    def x_taylor_s2(self, known, unknown, x):
        """ 2nd Order x_Taylor Series of s2 """
        return known[9] \
               + known[10]*x[0] \
               + known[11]*x[1] \
               + unknown[3]*x[0]**2 \
               + unknown[4]*x[0]*x[1] \
               + unknown[5]*x[1]**2
               
    def x_taylor_u(self, known, unknown, s, x):
        """ 4th Order x_taylor Series of u"""
        u = self.s_taylor_u(known, unknown, s)
        s1 = self.x_taylor_s1(known, unknown, x)
        s2 = self.x_taylor_s2(known, unknown, x)
        u = u.subs([(s[0], s1), (s[1], s2)])
        return u
    
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
        
        ds1_dx1 = ds_dx[0, 0]
        ds1_dx2 = ds_dx[0, 1]
        ds2_dx1 = ds_dx[1, 0]
        ds2_dx2 = ds_dx[1, 1]
               
        det = ds1_dx1*ds2_dx2 - ds1_dx2*ds2_dx1
        
        coeff_0_det = lambdify(x, det, 'numpy')
        coeff_1_det = lambdify(x, diff(det, x[0]), 'numpy')
        coeff_2_det = lambdify(x, diff(det, x[1]), 'numpy')
        coeff_0_det = coeff_0_det(x_value[0], x_value[1])
        coeff_1_det = coeff_1_det(x_value[0], x_value[1])
        coeff_2_det = coeff_2_det(x_value[0], x_value[1])
        
        coeff_0_ds1_dx1 = lambdify(x, ds1_dx1, 'numpy')
        coeff_1_ds1_dx1 = lambdify(x, diff(ds1_dx1, x[0]), 'numpy')
        coeff_2_ds1_dx1 = lambdify(x, diff(ds1_dx1, x[1]), 'numpy')
        coeff_0_ds1_dx1 = coeff_0_ds1_dx1(x_value[0], x_value[1])
        coeff_1_ds1_dx1 = coeff_1_ds1_dx1(x_value[0], x_value[1])
        coeff_2_ds1_dx1 = coeff_2_ds1_dx1(x_value[0], x_value[1])
        
        coeff_0_ds1_dx2 = lambdify(x, ds1_dx2, 'numpy')
        coeff_1_ds1_dx2 = lambdify(x, diff(ds1_dx2, x[0]), 'numpy')
        coeff_2_ds1_dx2 = lambdify(x, diff(ds1_dx2, x[1]), 'numpy')
        coeff_0_ds1_dx2 = coeff_0_ds1_dx2(x_value[0], x_value[1])
        coeff_1_ds1_dx2 = coeff_1_ds1_dx2(x_value[0], x_value[1])
        coeff_2_ds1_dx2 = coeff_2_ds1_dx2(x_value[0], x_value[1])
        
        coeff_0_ds2_dx1 = lambdify(x, ds2_dx1, 'numpy')
        coeff_1_ds2_dx1 = lambdify(x, diff(ds2_dx1, x[0]), 'numpy')
        coeff_2_ds2_dx1 = lambdify(x, diff(ds2_dx1, x[1]), 'numpy')
        coeff_0_ds2_dx1 = coeff_0_ds2_dx1(x_value[0], x_value[1])
        coeff_1_ds2_dx1 = coeff_1_ds2_dx1(x_value[0], x_value[1])
        coeff_2_ds2_dx1 = coeff_2_ds2_dx1(x_value[0], x_value[1])
        
        coeff_0_ds2_dx2 = lambdify(x, ds2_dx2, 'numpy')
        coeff_1_ds2_dx2 = lambdify(x, diff(ds2_dx2, x[0]), 'numpy')
        coeff_2_ds2_dx2 = lambdify(x, diff(ds2_dx2, x[1]), 'numpy')
        coeff_0_ds2_dx2 = coeff_0_ds2_dx2(x_value[0], x_value[1])
        coeff_1_ds2_dx2 = coeff_1_ds2_dx2(x_value[0], x_value[1])
        coeff_2_ds2_dx2 = coeff_2_ds2_dx2(x_value[0], x_value[1])
        
        coeff_0_dx1_ds1 = coeff_0_ds2_dx2/coeff_0_det
        coeff_1_dx1_ds1 = coeff_0_ds2_dx2/coeff_0_det \
                          *(coeff_1_ds2_dx2/coeff_0_ds2_dx2 \
                            - coeff_1_det/coeff_0_det)       
        coeff_2_dx1_ds1 = coeff_0_ds2_dx2/coeff_0_det \
                          *(coeff_2_ds2_dx2/coeff_0_ds2_dx2 \
                            - coeff_2_det/coeff_0_det) 
                          
        coeff_0_dx1_ds2 = - coeff_0_ds1_dx2/coeff_0_det
        coeff_1_dx1_ds2 = - coeff_0_ds1_dx2/coeff_0_det \
                          *(coeff_1_ds1_dx2/coeff_0_ds1_dx2 \
                            - coeff_1_det/coeff_0_det)       
        coeff_2_dx1_ds2 = coeff_0_ds1_dx2/coeff_0_det \
                          *(coeff_2_ds1_dx2/coeff_0_ds1_dx2 \
                            - coeff_2_det/coeff_0_det)
                          
        coeff_0_dx2_ds1 = - coeff_0_ds2_dx2/coeff_0_det
        coeff_1_dx2_ds1 = - coeff_0_ds2_dx1/coeff_0_det \
                          *(coeff_1_ds2_dx1/coeff_0_ds2_dx1 \
                            - coeff_1_det/coeff_0_det)       
        coeff_2_dx2_ds1 = coeff_0_ds2_dx1/coeff_0_det \
                          *(coeff_2_ds2_dx1/coeff_0_ds1_dx2 \
                            - coeff_2_det/coeff_0_det)
                          
        coeff_0_dx2_ds2 = coeff_0_ds1_dx1/coeff_0_det
        coeff_1_dx2_ds2 = coeff_0_ds1_dx1/coeff_0_det \
                          *(coeff_1_ds1_dx1/coeff_0_ds1_dx1 \
                            - coeff_1_det/coeff_0_det)       
        coeff_2_dx2_ds2 = coeff_0_ds1_dx1/coeff_0_det \
                          *(coeff_2_ds1_dx1/coeff_0_ds1_dx1 \
                            - coeff_2_det/coeff_0_det)
        
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
    
    def term_modified_x_taylor_laplacian_u(self, known, unknown, s, x):
        """ 1st Order x_Taylor Series of Laplacian of u """
        """ 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
        x_value = self.x_values[0][0]
        # u,1
        du_ds1 = self.x_taylor_du_ds(known, unknown, s, x)[0]
        # u,11
        ddu_dds1 = self.x_taylor_ddu_dds(known, unknown, s, x)[0, 0]
        # g11
        g11 = self.x_taylor_submetric(known, unknown, x)[0, 0]
        # g22
        g22 = self.x_taylor_submetric(known, unknown, x)[1, 1]
        # g11,1
        modified_dg11_ds1 = self.modified_x_taylor_dg11_ds1(known, unknown, x)
        # g22,1
        modified_dg22_ds1 = self.modified_x_taylor_dg22_ds1(known, unknown, x)
        # 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1
        temp = 2*g11*g22*ddu_dds1 \
               + (g11*modified_dg22_ds1 - \
                  modified_dg11_ds1*g22)*du_ds1
        
        coeff_0_laplacian_u = lambdify(x, temp, 'numpy')
        coeff_0_laplacian_u = coeff_0_laplacian_u(x_value[0], x_value[1])

        test = []
        test = [coeff_0_laplacian_u]
        return test
    
#    def term_s_taylor_du_ds2(self, known, unknown, s):
#        du_ds2 = self.s_taylor_du_ds(known, unknown, s)[1]
#        test = []
#        for i in range(len(Poly(du_ds2, s).coeffs())):
#            temp = Poly(du_ds2, s).coeffs()[i]
#            test.append(temp)
#        return test
    
    def term_x_taylor_g12(self, known, unknown, x):
        g12 = self.x_taylor_submetric(known, unknown, x)[0, 1]
        test = []
        for i in range(len(Poly(g12, x).coeffs())):
            temp = Poly(g12, x).coeffs()[i]
            test.append(temp)
        return test

    def solution(self, known, unknown, s, x):
        u_theory = self.u_theory
        x_value = self.x_values[0][0]
        a_theory = self.a_theory[0][0]
        b_theory = self.b_theory[0][0]
        
        f1 = self.term_x_taylor_g12(known, unknown, x)[0]
        f2 = self.term_x_taylor_g12(known, unknown, x)[1]
        f3 = self.term_x_taylor_g12(known, unknown, x)[2]
        f4 = self.term_x_taylor_g12(known, unknown, x)[3]
        f5 = self.term_x_taylor_g12(known, unknown, x)[4]
        f6 = self.term_modified_x_taylor_laplacian_u(known, unknown, s, x)[0]
        f = (f1, f2, f3, f4, f5, f6)
        
#        unknown_init = ((1 + random.uniform(-1.0, 1.0)/10)*a_theory[3],
#                        (1 + random.uniform(-1.0, 1.0)/10)*a_theory[4],
#                        (1 + random.uniform(-1.0, 1.0)/10)*a_theory[5],
#                        (1 + random.uniform(-1.0, 1.0)/10)*b_theory[3],
#                        (1 + random.uniform(-1.0, 1.0)/10)*b_theory[4],
#                        (1 + random.uniform(-1.0, 1.0)/10)*b_theory[5]
#                        )
        unknown_init = (a_theory[3],
                        a_theory[4],
                        a_theory[5],
                        b_theory[3],
                        b_theory[4],
                        b_theory[5]
                        )
        
        a_experiment = np.ndarray((3,))
        a_experiment[0] = nsolve(f, unknown, unknown_init)[0]
        a_experiment[1] = nsolve(f, unknown, unknown_init)[1]
        a_experiment[2] = nsolve(f, unknown, unknown_init)[2]
        
        b_experiment = np.ndarray((3,))
        b_experiment[0] = nsolve(f, unknown, unknown_init)[3]
        b_experiment[1] = nsolve(f, unknown, unknown_init)[4]
        b_experiment[2] = nsolve(f, unknown, unknown_init)[5]
        
        x_taylor_u = self.x_taylor_u(known, unknown, s, x)
        u_experiment = lambdify([unknown, x], x_taylor_u, 'numpy')
        u_experiment = u_experiment((a_experiment[0],
                                     a_experiment[1],
                                     a_experiment[2],
                                     b_experiment[0],
                                     b_experiment[1],
                                     b_experiment[2]),
                                    (x_value[0],
                                     x_value[1])
                                    )
                                    
        u_theory = lambdify(x, u_theory, 'numpy')
        u_theory = u_theory(x_value[0], x_value[1])     
        
        result = np.ndarray((2,))
        result[0] = u_theory
        result[1] = u_experiment
        return result    

        

if __name__ == '__main__':
    
    t0 = time.time()
    
    theory = laplace_theory.Theoretical_Values()
    
    s = [Symbol('s1', real = True), 
         Symbol('s2', real = True)
         ]
    
    x = [Symbol('x1', real = True), 
         Symbol('x2', real = True)
         ]
    
    known = [theory.r_theory(x)[0][0][0],
             theory.r_theory(x)[0][0][1],
             theory.r_theory(x)[0][0][2],
             theory.r_theory(x)[0][0][3],
             theory.r_theory(x)[0][0][4],
             theory.r_theory(x)[0][0][5],
             theory.a_theory(x)[0][0][0],
             theory.a_theory(x)[0][0][1],
             theory.a_theory(x)[0][0][2],
             theory.b_theory(x)[0][0][0],
             theory.b_theory(x)[0][0][1],
             theory.b_theory(x)[0][0][2]
             ]
    
    unknown = [Symbol('a11', real = True),
               Symbol('a12', real = True),
               Symbol('a22', real = True),
               Symbol('b11', real = True),
               Symbol('b12', real = True),
               Symbol('b22', real = True)
               ]
    
    print('x_values = ', theory.x_values(x))
    print('')
    
    taylor = Taylor_Functions(known, unknown, s, x)

#    print('Modified x_Taylor Series of Laplacian of u = ')
#    for i in range(len(taylor.term_modified_x_taylor_laplacian_u(known, unknown, s, x))):
#        display(taylor.term_modified_x_taylor_laplacian_u(known, unknown, s, x)[i])    
#    print('')

    print('(u_theory, u_experiment) = ')
    print(taylor.solution(known, unknown, s, x))
    print('')
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
        
    










