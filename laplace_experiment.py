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

# For Symbolic Expression Displaying
from IPython.display import display



class Known(laplace_theory.Theory):
    """ Known Values """
    
    def __init__(self, x, s, x_target):
        self.Theory = laplace_theory.Theory(x, s, x_target)
        
    def known(self):
        a_theory = self.Theory.a_theory()
        b_theory = self.Theory.b_theory()
        r_theory = self.Theory.r_theory()
        
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
        a_theory = self.Theory.a_theory()
        b_theory = self.Theory.b_theory()
#        r_theory = self.Theory.r_theory()
        
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
    
        r = 1.0
        unknown_init = np.ndarray((len(unknown),))
        for i in range(len(unknown)):
            unknown_init[i] = (1 + random.uniform(-r, r)/100)*unknown_theory[i]
        
        return unknown_init


class Taylor(Known, Unknown):
    """ Taylor Series Expressions """
    
    def __init__(self, x, s, x_target, unknown):
        self.Known = Known(x, s, x_target)
        self.Unknown = Unknown(x, s, x_target, unknown)
        self.x_target = x_target
        
    def x_taylor_s(self, x, unknown):
        """ 2nd Order x_Taylor Series of s1 """
        x_target = self.x_target
        known = self.Known.known()
        
        x_taylor_s = np.ndarray((2,), 'object')
        
        x_taylor_s[0] = known[0] \
                        + known[1]*(x[0] - x_target[0]) \
                        + known[2]*(x[1] - x_target[1]) \
                        + unknown[0]*(x[0] - x_target[0])**2/2 \
                        + unknown[1]*(x[0] - x_target[0])*(x[1] - x_target[1]) \
                        + unknown[2]*(x[1] - x_target[1])**2/2
               
        x_taylor_s[1] = known[3] \
                        + known[4]*(x[0] - x_target[0]) \
                        + known[5]*(x[1] - x_target[1]) \
                        + unknown[3]*(x[0] - x_target[0])**2/2 \
                        + unknown[4]*(x[0] - x_target[0])*(x[1] - x_target[1]) \
                        + unknown[5]*(x[1] - x_target[1])**2/2
        
        return x_taylor_s
        
    def s_target(self, unknown_init):
        x_target = self.x_target
        s_target = self.x_taylor_s(x_target, unknown_init)
        
        return s_target
        
    def s_taylor_u(self, s, unknown):
        """ 2nd Order x_Taylor Series of s1 """
        known = self.Known.known()
        s_target = self.s_target(unknown_init)
        
        s_taylor_u = known[6] \
                     + known[7]*(s[0] - s_target[0]) \
                     + known[8]*(s[1] - s_target[1]) \
                     + known[9]*(s[0] - s_target[0])**2/2 \
                     + known[10]*(s[0] - s_target[0])*(s[1] - s_target[1]) \
                     + known[11]*(s[1] - s_target[1])**2/2
                     
        return s_taylor_u
    
    def x_taylor_u(self, x, unknown):
        """ Convolution of x_taylor_s & s_taylor_u """
        x_taylor_s = self.x_taylor_s(x, unknown)
        x_taylor_u = self.s_taylor_u(x_taylor_s, unknown)
        
        return x_taylor_u
    
 
class BoundaryConditions(Taylor):
    """ Boundary s_coordinate """
    
    def __init__(self, x, s, x_target, unknown):
        self.Taylor = Taylor(x, s, x_target, unknown)
        self.Unknown = self.Taylor.Unknown
        self.x = x
        self.x_target = x_target
        self.unknown = unknown
        
    def s_boundary(self, unknown_init):
        s_target = self.Taylor.s_target(unknown_init)
        s_boundary = np.ndarray((2, len(s_target)))
        
        s_boundary[0][0] = s_target[0] - 1.0
        s_boundary[0][1] = s_target[1]
        s_boundary[1][0] = s_target[0] + 1.0
        s_boundary[1][1] = s_target[1]
        
        return s_boundary
    
    def x_boundary(self, unknown_init):
        x = self.x
        x_target = self.x_target
        x_taylor_s = self.Taylor.x_taylor_s(x, unknown_init)
        s_boundary = self.s_boundary(unknown_init)
        
        f = np.ndarray((2, len(x),), 'object')
        for i in range(2):
            for j in range(len(x)):
                f[i][j] = x_taylor_s[j] - s_boundary[i][j]

        x_boundary = np.ndarray((2, len(x),))
        for i in range(2):
            for j in range(len(x)):
                x_boundary[i][j] = nsolve((f[i][0], f[i][1]), \
                                          (x[0], x[1]), \
                                          (x_target[0], x_target[1]) \
                                          )[j]
                
        return x_boundary
    
    def u_boundary(self, unknown_init):
        x_boundary = self.x_boundary(unknown_init)
        
        u_boundary = np.ndarray((2),)
        for i in range(2):
            u_boundary[i] = self.Taylor.x_taylor_u(x_boundary[i], unknown_init)
        
        return u_boundary
    
    def boundary_conditions(self):
        unknown = self.unknown
        x_boundary = self.x_boundary(unknown_init)
        u_boundary = self.u_boundary(unknown_init)
        
        u = np.ndarray((2,), 'object')
        bc = np.ndarray((2,), 'object')
        for i in range(2):
            u[i] = self.Taylor.x_taylor_u(x_boundary[i], unknown)
            bc[i] = u[i] - u_boundary[i]
        
        return bc
        

class Derivative(Taylor):
    """ x_Taylor Series of Derivatives """
    
    def __init__(self, x, s, x_target, unknown):
        self.Taylor = Taylor(x, s, x_target, unknown)
        self.x = x
        self.s = s
        self.unknown = unknown
        
    def du_ds(self):
        x = self.x
        s = self.s
        unknown = self.unknown
        s_taylor_u = self.Taylor.s_taylor_u(s, unknown)
        x_taylor_s = self.Taylor.x_taylor_s(x, unknown)
        
        du_ds = np.ndarray((2,), 'object')
        for i in range(2):
            du_ds[i] = diff(s_taylor_u, s[i])
        
        for i in range(2):
            du_ds[i] = lambdify(s, du_ds[i], 'numpy')
            du_ds[i] = du_ds[i](x_taylor_s[0], x_taylor_s[1])
        
        return du_ds
                    
    def ddu_dds(self):
        x = self.x
        s = self.s
        unknown = self.unknown
        s_taylor_u = self.Taylor.s_taylor_u(s, unknown)
        x_taylor_s = self.Taylor.x_taylor_s(x, unknown)
        
        ddu_dds = np.ndarray((2, 2), 'object')
        ddu_dds[0][0] = diff(s_taylor_u, s[0], 2)
        ddu_dds[0][1] = diff(s_taylor_u, s[0], s[1])
        ddu_dds[1][0] = diff(s_taylor_u, s[1], s[0])
        ddu_dds[1][1] = diff(s_taylor_u, s[1], 2)
        
        for i in range(2):
            for j in range(2):
                ddu_dds[i][j] = lambdify(s, ddu_dds[i][j], 'numpy')
                ddu_dds[i][j] = ddu_dds[i][j](x_taylor_s[0], x_taylor_s[1])       
        
        return ddu_dds
       
    def ds_dx(self):
        x = self.x
        unknown = self.unknown
        x_taylor_s = self.Taylor.x_taylor_s(x, unknown)
        
        ds_dx = np.ndarray((2, 2,), 'object')
        ds_dx[0][0] = diff(x_taylor_s[0], x[0])
        ds_dx[0][1] = diff(x_taylor_s[0], x[1])
        ds_dx[1][0] = diff(x_taylor_s[1], x[0])
        ds_dx[1][1] = diff(x_taylor_s[1], x[1])
                
        return ds_dx
    
    def dx_ds(self):
        """ Inverse Matrix Library NOT Used due to High Computational Cost """
        ds_dx = self.ds_dx()
        det = ds_dx[0][0]*ds_dx[1][1] - ds_dx[0][1]*ds_dx[1][0]
        
        dx_ds = np.ndarray((2, 2), 'object')
        dx_ds[0][0] = ds_dx[1][1]/det
        dx_ds[0][1] = - ds_dx[0][1]/det
        dx_ds[1][0] = - ds_dx[1][0]/det
        dx_ds[1][1] = ds_dx[0][0]/det
    
        return dx_ds


class Metric(Derivative):
    """ x_Taylor Series of Metrics """
    
    def __init__(self, x, s, x_target, unknown):
        self.Derivative = Derivative(x, s, x_target, unknown)
        self.x = x
    
    def submetric(self):
        """ Subscript Metric g_ij """
        ds_dx = self.Derivative.ds_dx()
        
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
  
    def dg_ds1(self):
        """ dg_ij/ds1 = dx1/ds1*dg_ij/dx1 + dx2/ds1*dg_ij/dx2 """
        dx1_ds1 = self.Derivative.dx_ds()[0][0]
        dx2_ds1 = self.Derivative.dx_ds()[1][0]
        submetric = self.submetric()
        x = self.x
        
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
    
    def __init__(self, x, s, x_target, unknown):
        self.Metric = Metric(x, s, x_target, unknown)
        self.Derivative = self.Metric.Derivative
        self.x =  x
        
    def governing_equation_1(self):
        """ 1st Order x_Taylor Series of g_12 """
        g12 = self.Metric.submetric()[0][1]
        x = self.x
        
        coeff_g12 = np.ndarray((len(x) + 1,), 'object')
        coeff_g12[0] = diff(g12, x[0])
        coeff_g12[1] = diff(g12, x[1])
        coeff_g12[2] = g12
        
        for i in range(len(x) + 1):
            coeff_g12[i] = lambdify(x, coeff_g12[i], 'numpy')
            coeff_g12[i] = coeff_g12[i](0, 0)
         
        return coeff_g12
    
    def governing_equation_2(self):
        """ 1st Order x_Taylor Series of Laplacian of u """
        """ 2*g11*g22*u,11 + (g11*g22,1 - g11,1*g22)*u,1 """
        du_ds1 = self.Derivative.du_ds()[0]
        ddu_dds1 = self.Derivative.ddu_dds()[0][0]
        g11 = self.Metric.submetric()[0][0]
        g22 = self.Metric.submetric()[1][1]
        dg11_ds1 = self.Metric.dg_ds1()[0][0]
        dg22_ds1 = self.Metric.dg_ds1()[1][1]
        x = self.x
        
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
    
    def __init__(self, x, s, x_target, unknown):
        self.BC = BoundaryConditions(x, s, x_target, unknown)
        self.GE = GoverningEquations(x, s, x_target, unknown)
        self.Unknown = self.BC.Taylor.Unknown
        self.unknown = unknown
    
    def f(self):
        unknown = self.unknown
        bc = self.BC.boundary_conditions()
        ge1 = self.GE.governing_equation_1()
        ge2 = self.GE.governing_equation_2()
        
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
            b[i] = -f[i]
            for j in range(len(unknown)):
                b[i] += diff(f[i], unknown[j])*unknown[j]
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
    
    def error_norm(self, unknown_temp):
        unknown = self.unknown
        f = self.f()
        
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
    
    def solution(self, unknown_init):
        unknown_temp = unknown_init
        error = self.error_norm(unknown_temp)
        
        while error > 1.0e-8:
            A = self.A(unknown_temp)
            b = self.b(unknown_temp)
            unknown_temp = solve(A, b)        
            error = self.error_norm(unknown_temp)
        
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
    
    x_target_array = np.ndarray((n, n, len(x),))
    
    unknown_theory_array = np.ndarray((n, n, len(unknown)))
    unknown_init_array = np.ndarray((n, n, len(unknown)))
    unknown_experiment_array = np.ndarray((n, n, len(unknown)))
    
    error_init_array = np.ndarray((n, n, 1))
    error_experiment_array = np.ndarray((n, n, 1))
    
    eigvals_A_init_array = np.ndarray((n, n, len(unknown)), 'complex')
    
    
    def error(a, b):
        
        error = round(norm(b - a)/norm(a), 4)*100
        
        return error
    
    
    for i in range(n):
        for j in range(n):
            x_target[0] = 1.0 + i/n
            x_target[1] = 1.0 + j/n
            
            ################################################
            Unknown_call = Unknown(x, s, x_target, unknown)
            ################################################
            unknown_theory = Unknown_call.unknown_theory()
            unknown_init = Unknown_call.unknown_init()
            
            ######################################################
            Experiment_call = Experiment(x, s, x_target, unknown)
            ######################################################
            unknown_experiment = Experiment_call.solution(unknown_init)
            A_init = Experiment_call.A(unknown_init)
            eigvals_A_init = eigvals(A_init)
            
            error_init = error(unknown_theory, unknown_init)
            error_experiment = error(unknown_theory, unknown_experiment)
            
            for k in range(len(x)):
                x_target_array[i][j][k] = x_target[k]
    
            for k in range(len(unknown)):
                unknown_theory_array[i][j][k] = unknown_theory[k]
                unknown_init_array[i][j][k] = unknown_init[k]
                unknown_experiment_array[i][j][k] = unknown_experiment[k]
                eigvals_A_init_array[i][j][k] = eigvals_A_init[k]
                
            for k in range(1):
                error_init_array[i][j][k] = error_init
                error_experiment_array[i][j][k] = error_experiment
        
    print('unknown_theory = ')
    print(unknown_theory_array)
    print('')
            
    print('unknown_init = ')
    print(unknown_init_array)
    print('')
          
    print('error_init = ')
    print(error_init_array)
    print()    
    print('unknown_experiment = ')
    print(unknown_experiment_array)
    print('')
          
    print('error_experiment = ')
    print(error_experiment_array)
    print('')
    
    print('eigvals_A_init = ')
    print(eigvals_A_init_array)
    print('')
    
    
    t1 = time.time()
    
    print('Elapsed Time = ')
    print(round(t1 - t0), '(s)')
    print('')






        
    









