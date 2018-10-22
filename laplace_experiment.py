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
from numpy.linalg import norm, solve, eigvals

# For Symbolic Notation
import sympy as sym
from sympy import Symbol, diff, lambdify, nsolve
sym.init_printing()

# For Random Variables
import random

# For Measuring Computation Time
import time



class Known(laplace_theory.Theory):
    """ Known Values """
    
    def __init__(self, x, s, x_target):
        self.Theory = laplace_theory.Theory(x, s, x_target)
        
    def known(self):
        a_theory = self.Theory.a_theory(x_target)
        b_theory = self.Theory.b_theory(x_target)
        r_theory = self.Theory.r_theory(x_target)
        
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
        
        return known


class Unknown(laplace_theory.Theory):
    """ Unknowns Related """

    def __init__(self, x, s, x_target, unknown):
        self.Theory = laplace_theory.Theory(x, s, x_target)
        self.unknown = unknown
        
    def unknown_theory(self):
        unknown = self.unknown
        a_theory = self.Theory.a_theory(x_target)
        b_theory = self.Theory.b_theory(x_target)
#        r_theory = self.Theory.r_theory(x_target)
        
        unknown_theory = np.ndarray((len(unknown),))
        unknown_theory[0] = a_theory[3]
        unknown_theory[1] = a_theory[4]
        unknown_theory[2] = a_theory[5]
        unknown_theory[3] = b_theory[3]
        unknown_theory[4] = b_theory[4]
        unknown_theory[5] = b_theory[5]
        
        return unknown_theory
        
    def unknown_init(self):
        unknown_theory = self.unknown_theory()
    
        r = 0.0
        unknown_init = np.ndarray((len(unknown),))
        for i in range(len(unknown)):
            unknown_init[i] = (1 + random.uniform(-r, r)/100)*unknown_theory[i]
        
#        unknown_init = ((1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[0],
#                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[1],
#                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[2],
#                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[3],
#                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[4],
#                        (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[5]
#                        )
        
        return unknown_init


class X_Taylor_S(Known):
    """ x_Taylor Series Expressions of s """
    
    def __init__(self, x, s, x_target, unknown):
        self.Known = Known(x, s, x_target)
        self.x_target = x_target
        
        
    def x_taylor_s(self, x, unknown):
        """ 2nd Order x_Taylor Series of s1 """
        x_target = self.x_target
        known = self.Known.known()
        
        s = np.ndarray((2,), 'object')
        
        s[0] = known[0] \
               + known[1]*(x[0] - x_target[0]) \
               + known[2]*(x[1] - x_target[1]) \
               + unknown[0]*(x[0] - x_target[0])**2/2 \
               + unknown[1]*(x[0] - x_target[0])*(x[1] - x_target[1]) \
               + unknown[2]*(x[1] - x_target[1])**2/2
               
        s[1] = known[3] \
               + known[4]*(x[0] - x_target[0]) \
               + known[5]*(x[1] - x_target[1]) \
               + unknown[3]*(x[0] - x_target[0])**2/2 \
               + unknown[4]*(x[0] - x_target[0])*(x[1] - x_target[1]) \
               + unknown[5]*(x[1] - x_target[1])**2/2
        
        return s
    

class S_Taylor_U(Known, S_Target):
    """ S_Taylor Series Expressions of u """
    
    def __init__(self, x, s, x_target, unknown):
        self.Known = Known(x, s, x_target)
        self.S_Target = S_Target(x, s, x_target, unknown)
        
    def s_taylor_u(self, s, unknown):
        """ 2nd Order x_Taylor Series of s1 """
        s_target = self.S_Target.s_target()
        known = self.Known.known()
#        unknown = self.unknown
        
        u = known[6] \
            + known[7]*(s[0] - s_target[0]) \
            + known[8]*(s[1] - s_target[1]) \
            + known[9]*(s[0] - s_target[0])**2/2 \
            + known[10]*(s[0] - s_target[0])*(s[1] - s_target[1]) \
            + known[11]*(s[1] - s_target[1])**2/2
    
        return u
    

class X_Taylor_U(S_Taylor_U, X_Taylor_S):
    """ X_Taylor Series of u """

    def __init__(self, x, s, x_target, unknown):
        self.S_Taylor_U = S_Taylor_U(x, s, x_target, unknown)
        self.X_Taylor_S = X_Taylor_S(x, s, x_target, unknown)
        
    def x_taylor_u(self, x, unknown):
        x_taylor_s = self.X_Taylor_S.x_taylor_s(x, unknown)
        u = self.S_Taylor_U.s_taylor_u(x_taylor_s, unknown)
        
        return u
    

class S_Target(X_Taylor_S, Unknown):
    """ Target s_coordinate """
    
    def __init__(self, x, s, x_target, unknown):
        self.X_Taylor_S = X_Taylor_S(x, s, x_target, unknown)
        self.Unknown = Unknown(x, s, x_target, unknown)
        
    def s_target(self):
        unknown_init = self.Unknown.unknown_init()
        s_target = self.X_Taylor_S.x_taylor_s(x_target, unknown_init)
        
        return s_target
    
 
class Boundary(S_Target, X_Taylor_U):
    """ Boundary s_coordinate """
    
    def __init__(self, x, s, x_target, unknown):
        self.S_Target = S_Target(x, s, x_target, unknown)
        self.X_Taylor_S = self.S_Target.X_Taylor_S
        self.Unknown = self.S_Target.Unknown
        self.X_Taylor_U = X_Taylor_U(x, s, x_target, unknown)
        self.x = x
        self.x_target = x_target
        
    def s_boundary(self):
        s_target = self.S_Target.s_target()
        s_boundary = np.ndarray((2, len(s_target)))
        
        s_boundary[0][0] = s_target[0] - 1.0
        s_boundary[0][1] = s_target[1]
        s_boundary[1][0] = s_target[0] + 1.0
        s_boundary[1][1] = s_target[1]
        
        return s_boundary
    
    def x_boundary(self):
        x = self.x
        x_target = self.x_target
        unknown_init = self.Unknown.unknown_init()
        s1 = self.X_Taylor_S.x_taylor_s(x, unknown_init)[0]
        s2 = self.X_Taylor_S.x_taylor_s(x, unknown_init)[1]
        s_boundary = self.s_boundary()
        
        f = np.ndarray((2, len(x),), 'object')
        for i in range(2):
            f[i][0] = s1 - s_boundary[i][0]
            f[i][1] = s2 - s_boundary[i][1]
        
        x_boundary = np.ndarray((2, 2,))
        for i in range(2):
            for j in range(2):
                x_boundary[i][j] = nsolve((f[i][0], f[i][1]), \
                                          (x[0], x[1]), \
                                          (x_target[0], x_target[1]) \
                                          )[j]
                
        return x_boundary
    
    def u_boundary(self):
        unknown_init = self.Unknown.unknown_init()
        x_boundary = self.x_boundary()
        
        u_boundary = np.ndarray((2),)
        for i in range(2):
            u_boundary[i] = self.X_Taylor_U.x_taylor_u(x_boundary[i], unknown_init)
        
        return u_boundary
        
    
class RelativeErrorNorm():
    """ Relative Error Norm of Vector """
    
    def relative_error_norm(self, a, b):
        
        relative_error_norm = round(norm(b - a)/norm(a), 4)*100
        
        return relative_error_norm
    
    
    
    
    
    
    
    

    




    
class BoundaryConditions(Boundary):
    """ Derive Boundary Conditions """
    
    def __init__(self, x_boundary, u_boundary):
        self.boundary = Boundary(s_value)
        self.u_boundary = u_boundary    
    
    def bc(self):
        u = self.boundary.taylor.x_taylor_u()
        
        bc = np.ndarray((2,), 'object')
        for i in range(2):
            bc[i] = u - u_boundary[i]
            bc[i] = lambdify(x, bc[i], 'numpy')
            bc[i] = bc[i](x_boundary[i][0], x_boundary[i][1])
        
        return bc
        

class Derivative(TaylorExpansion):
    """ x_Taylor Series of Derivatives """
    
    def __init__(self, x, s):
        self.taylor = TaylorExpansion(x, s, x_target, s_target, known, unknown)
        self.x = x
        self.s = s
        
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


class Metric(Derivative):
    """ x_Taylor Series of Metrics """
    
    def __init__(self, x):
        self.derivative = Derivative(x, s)
        self.x = x
    
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
        
    
class GoverningEquations(Metric):
    """ Derive Governing Equations """
    
    def __init__(self, x_target, s_target, known, unknown_init):
        self.metric = Metric(x)
        self.unknown_init = unknown_init
        
    def ge1(self):
        """ 1st Order x_Taylor Series of g_12 """
        g12 = self.metric.x_taylor_submetric()[0][1]
        
        coeff_g12 = np.ndarray((len(x) + 1,), 'object')
        coeff_g12[0] = diff(g12, x[0])
        coeff_g12[1] = diff(g12, x[1])
        coeff_g12[2] = g12
        
        for i in range(len(x) + 1):
            coeff_g12[i] = lambdify(x, coeff_g12[i], 'numpy')
            coeff_g12[i] = coeff_g12[i](0, 0)
         
        return coeff_g12
    
    def ge2(self):
        """ 1st Order x_Taylor Series of Laplacian of u """
        """ 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
        du_ds1 = self.metric.derivative.x_taylor_du_ds()[0]
        ddu_dds1 = self.metric.derivative.x_taylor_ddu_dds()[0][0]
        g11 = self.metric.x_taylor_submetric()[0][0]
        g22 = self.metric.x_taylor_submetric()[1][1]
        dg11_ds1 = self.metric.x_taylor_dg_ds1()[0][0]
        dg22_ds1 = self.metric.x_taylor_dg_ds1()[1][1]
        
        laplacian_u = 2*g11*g22*ddu_dds1 \
                      + (g11*dg22_ds1 - g22*dg11_ds1)*du_ds1
        
        coeff_laplacian_u = np.ndarray((len(x) + 1,), 'object')
        coeff_laplacian_u[0] = diff(laplacian_u, x[0])
        coeff_laplacian_u[1] = diff(laplacian_u, x[1])
        coeff_laplacian_u[2] = laplacian_u 
        
        for i in range(len(x) + 1):
            coeff_laplacian_u[i] = lambdify(x, coeff_laplacian_u[i], 'numpy')
            coeff_laplacian_u[i] = coeff_laplacian_u[i](0, 0)
         
        return coeff_laplacian_u
    

class Experiment(BoundaryConditions, GoverningEquations):
    """ Solve G.E. & B.C. """
    
    def __init__(self, x_target, s_target, known, unknown, unknown_init):
        self.bc = BoundaryConditions(x_boundary, u_boundary)
        self.ge = GoverningEquations(x_target, s_target, known, unknown)
        self.unknown = unknown
        self.unknown_init = unknown_init
    
    def f(self):
        unknown = self.unknown
        bc = self.bc.bc()
        ge1 = self.ge.ge1()
        ge2 = self.ge.ge2()
        
        f = np.ndarray((len(unknown),), 'object')
        f[0] = bc[0]
        f[1] = bc[1]
        f[2] = ge1[0]
        f[3] = ge1[1]
        f[4] = ge2[0]
        f[5] = ge2[1]
        
        return f
    
    def A(self, unknown_temp):
        unknown = self.unknown
        f = self.f()
        
        A = np.ndarray((len(f), len(unknown),), 'object')
        for i in range(len(f)):
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
        unknown = self.unknown
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
    
    def solution(self):
        unknown = self.unknown
        f = self.f()
        unknown_temp = self.unknown_init
        
        def error_norm(unknown_temp):
            error = np.ndarray((len(f),), 'object')
            for i in range(len(f)):
                error[i] = lambdify(unknown, f[i], 'numpy')
                error[i] = error[i](unknown_temp[0],
                                    unknown_temp[1],
                                    unknown_temp[2],
                                    unknown_temp[3],
                                    unknown_temp[4],
                                    unknown_temp[5]
                                    )
            error_norm = norm(error)
            
            return error_norm
        
        error = error_norm(unknown_temp)
        
        while error > 1.0e-8:
            A = self.A(unknown_temp)
            b = self.b(unknown_temp)
            unknown_temp = solve(A, b)        
            error = error_norm(unknown_temp)
        
        solution = unknown_temp
        
        return solution

        

if __name__ == '__main__':
    
    t0 = time.time()
    
    x = np.ndarray((2,), 'object')
    x[0] = Symbol('x1', real = True)
    x[1] = Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = Symbol('s1', real = True)
    s[1] = Symbol('s2', real = True)
    
    unknown = np.ndarray((6,), 'object')
    unknown[0] = Symbol('a11', real = True)
    unknown[1] = Symbol('a12', real = True)
    unknown[2] = Symbol('a22', real = True)
    unknown[3] = Symbol('b11', real = True)
    unknown[4] = Symbol('b12', real = True)
    unknown[5] = Symbol('b22', real = True)
    
    n = 1
    
    x_target = np.ndarray((len(x),))
    for i in range(n):
        for j in range(n):
            x_target[0] = 1.0 + i/n
            x_target[1] = 1.0 + j/n
            
            ##########################################
            test = Unknown(x, s, x_target, unknown)
            ##########################################
            unknown_theory = test.unknown_theory()
            unknown_init = test.unknown_init()
            
            #####################################################
            Boundary = Boundary(x, s, x_target, unknown)
            #####################################################
            s_boundary = Boundary.s_boundary()
            x_boundary = Boundary.x_boundary()
            u_boundary = Boundary.u_boundary()
            
            ####################################################################
            Error = RelativeErrorNorm()
            ####################################################################
            error_init = Error.relative_error_norm(unknown_theory, unknown_init)
    
    print('unknown_theory = ')
    print(unknown_theory)
    print()
            
    print('unknown_init = ')
    print(unknown_init)
    print()
          
    print('error_init = ')
    print(error_init)
    print()

    print('s_boundary = ')
    print(s_boundary)
    print('')
    
    print('x_boundary = ')
    print(x_boundary)
    print('')

    print('u_boundary = ')
    print(u_boundary)
    print('')
    
    
    
#    #####################################
#    function = laplace_theory.Function()
#    #####################################
#    x = np.ndarray((2,), 'object')
#    x[0] = Symbol('x1', real = True)
#    x[1] = Symbol('x2', real = True)
#    
#    s = np.ndarray((2,), 'object')
#    s[0] = Symbol('s1', real = True)
#    s[1] = Symbol('s2', real = True)
#    
#    x_target = np.ndarray((len(x),))
#    s_target = np.ndarray((len(s),))
#    
#    x_boundary = np.ndarray((2, len(x),))
#    s_boundary = np.ndarray((2, len(s),))
#    u_boundary = np.ndarray((2, 1,))
#    
#    a = function.a(x)
#    b = function.b(x)
#    r = function.r(s)
#    
#    known = np.ndarray((12,))
#    unknown = np.ndarray((18 - len(known),), 'object')
#    
#    n = 1
#    
#    x_target_array = np.ndarray((n, n, len(x),))
#    s_target_array = np.ndarray((n, n, len(s)))
#    
#    x_boundary_array = np.ndarray((n, n, 2, len(x),))
#    s_boundary_array = np.ndarray((n, n, 2, len(s),))
#    u_boundary_array = np.ndarray((n, n, 2, 1,))
#    
#    a_theory_array = np.ndarray((n, n, len(a),))
#    b_theory_array = np.ndarray((n, n, len(b),))
#    r_theory_array = np.ndarray((n, n, len(r),))
#    
#    unknown_theory = np.ndarray(len(unknown),)
#    unknown_init = np.ndarray(len(unknown),)
#    unknown_experiment = np.ndarray(1,)
#    
#    known_array = np.ndarray((n, n, len(known),))
#    unknown_theory_array = np.ndarray((n, n, len(unknown),))
#    unknown_init_array = np.ndarray((n, n, len(unknown),))
#    unknown_experiment_array = np.ndarray((n, n, len(unknown),))
#    
#    error_init_array = np.ndarray((n, n, 1))
#    error_experiment_array = np.ndarray((n, n, 1))
#    
#    A_init_array = np.ndarray((n, n, len(unknown), len(unknown),))
#    b_init_array = np.ndarray((n, n, len(unknown),))
#    eigvals_A_init_array = np.ndarray((n, n, len(unknown),), 'complex')
#    
#    ##############################################
#    theory = laplace_theory.Theory(x, s, x_target)
#    ##############################################
#    def relative_error_norm(b, a):
#        relative_error_norm = round(norm(b - a)/norm(a), 4)*100
#        
#        return relative_error_norm
#    
#    for i in range(n):
#        for j in range(n):
#            x_target[0] = 1.0 + i/n
#            x_target[1] = 1.0 + j/n
#            
#            a_theory = theory.a_theory()
#            b_theory = theory.b_theory()
#            r_theory = theory.r_theory()
#            
#            s_target[0] = a_theory[0]
#            s_target[1] = b_theory[0]
#            
#            s_boundary[0][0] = s_target[0] - 1.0
#            s_boundary[0][1] = s_target[1] 
#            
#            s_boundary[1][0] = s_target[0] + 1.0
#            s_boundary[1][1] = s_target[1] 
#            
#            for k in range(2):
#                u_boundary[k] = function.u(s_boundary[k])
#            
#            known[0] = a_theory[0]
#            known[1] = a_theory[1]
#            known[2] = a_theory[2]
#            known[3] = b_theory[0]
#            known[4] = b_theory[1]
#            known[5] = b_theory[2]
#            known[6] = r_theory[0]
#            known[7] = r_theory[1]
#            known[8] = r_theory[2]
#            known[9] = r_theory[3]
#            known[10] = r_theory[4]
#            known[11] = r_theory[5]
#            
#            unknown_theory[0] = a_theory[3]
#            unknown_theory[1] = a_theory[4]
#            unknown_theory[2] = a_theory[5]
#            unknown_theory[3] = b_theory[3]
#            unknown_theory[4] = b_theory[4]
#            unknown_theory[5] = b_theory[5]
#            
#            unknown_init = ((1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[0],
#                            (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[1],
#                            (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[2],
#                            (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[3],
#                            (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[4],
#                            (1 + random.uniform(-0.0, 0.0)/100)*unknown_theory[5]
#                            )
#            error_init = relative_error_norm(unknown_init, unknown_theory)
#            
#            for k in range(len(unknown)):
#                unknown[k] = unknown_init[k]
#            
#            ##############################################
#            boundary1 = Boundary(s_boundary[0])
#            ##############################################
#            
#            x_boundary[0] = boundary1.solution()
#            
#            ##############################################
#            boundary2 = Boundary(s_boundary[1])
#            ##############################################
#            
#            x_boundary[1] = boundary2.solution()
#            
#            unknown = np.ndarray((18 - len(known),), 'object')
#            unknown[0] = Symbol('a11', real = True)
#            unknown[1] = Symbol('a12', real = True)
#            unknown[2] = Symbol('a22', real = True)
#            unknown[3] = Symbol('b11', real = True)
#            unknown[4] = Symbol('b12', real = True)
#            unknown[5] = Symbol('b22', real = True)
#            
#            ##########################################################################
#            experiment = Experiment(x_target, s_target, known, unknown, unknown_init)
#            ##########################################################################
#            unknown_experiment = experiment.solution()
#            error_experiment = relative_error_norm(unknown_experiment, unknown_theory)
#            
#            A_init = experiment.A(unknown_init)
#            b_init = experiment.b(unknown_init)
#            eigvals_A_init = eigvals(A_init)
#            
#            for k in range(len(unknown)):
#                if absolute(eigvals_A_init[k]) < 1.0e-4:
#                    eigvals_A_init[k] = 0
#                else:
#                    real = round(eigvals_A_init[k].real, 2)
#                    imag = round(eigvals_A_init[k].imag, 2)
#                    eigvals_A_init[k] = real + imag*1j
#                
#                for l in range(len(unknown)):
##                    if absolute(A_init[k][l]) < 1.0e-3:
##                        A_init[k][l] = 0
#                    A_init[k][l] = round(A_init[k][l], 4)
#            
#            for k in range(len(x)):
#                x_target_array[i][j][k] = x_target[k]
#                
#            for k in range(len(s)):
#                s_target_array[i][j][k] = s_target[k]
#            
#            for k in range(2):
#                x_boundary_array[i][j][k] = x_boundary[k]
#                s_boundary_array[i][j][k] = s_boundary[k]
#                u_boundary_array[i][j][k] = u_boundary[k]
#                
#            for k in range(len(a)):
#                a_theory_array[i][j][k] = a_theory[k]
#            
#            for k in range(len(b)):
#                b_theory_array[i][j][k] = b_theory[k]
#                
#            for k in range(len(r)):
#                r_theory_array[i][j][k] = r_theory[k]
#                
#            for k in range(len(known)):
#                known_array[i][j][k] = known[k]
#                
#            for k in range(len(unknown)):
#                unknown_theory_array[i][j][k] = unknown_theory[k]
#                unknown_init_array[i][j][k] = unknown_init[k]
#                unknown_experiment_array[i][j][k] = unknown_experiment[k]
#                
#                for l in range(len(unknown)):
#                    A_init_array[i][j][k][l] = A_init[k][l]
#                    
#                b_init_array[i][j][k] = b_init[k]
#                eigvals_A_init_array[i][j][k] = eigvals_A_init[k]
#                
#            for k in range(1):
#                error_init_array[i][j][k] = error_init
#                error_experiment_array[i][j][k] = error_experiment
#            
#    print('x_target = ')
#    print(x_target_array)
#    print('')
#    
#    print('x_boundary = ')
#    print(x_boundary_array)
#    print('')
#    
#    print('u_boundary = ')
#    print(u_boundary_array)
#    print('')
#    
#    print('unknown_theory = ')
#    print(unknown_theory_array)
#    print('')
#    
#    print('unknown_init = ')
#    print(unknown_init_array)
#    print('') 
#        
#    print('Error(%) of unknown_init = ')
#    print(error_init_array)
#    print('')
#
#    print('unknown_experiment = ')
#    print(unknown_experiment_array)
#    print('')
#    
#    print('Error(%) of unknown_experiment = ')
#    print(error_experiment_array)
#    print('')
#    
#    print('A_init = ')
#    print(A_init_array)
#    print('')
#    
#    print('Eigen Vaules of A_init = ')
#    print(eigvals_A_init_array)
#    print('')


    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')
        
    














