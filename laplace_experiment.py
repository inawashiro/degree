# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:00:27 2018

@author: inawashiro
"""
import laplace_theory

# For Numerical Computation 
import numpy as np
from numpy import dot
from numpy.linalg import norm, solve, eigvals, lstsq

# For Symbolic Notation
import sympy as syp
from sympy import Symbol, diff, lambdify, nsolve
syp.init_printing()

import scipy as scp
from scipy.sparse.linalg import spsolve, lsqr

# For Displaying Symbolic Notation
from IPython.display import display

# For Random Variables
import random

# For Measuring Computation Time
import time



class Known(laplace_theory.TheoreticalValue):
    """ Known Values """
    
    def __init__(self, f_id, x, s, unknown, x_value):
        self.Theory = laplace_theory.TheoreticalValue(f_id, x, s, x_value)
        self.unknown = unknown
        
    def known(self):
        s_coeff_theory = self.Theory.s_coeff_theory()
        u_coeff_theory = self.Theory.u_coeff_theory()
        unknown = self.unknown
        
        known = np.ndarray((18 - len(unknown),))
        known[0] = s_coeff_theory[0][0]
        known[1] = s_coeff_theory[0][1]
        known[2] = s_coeff_theory[0][2]
        known[3] = s_coeff_theory[1][0]
        known[4] = s_coeff_theory[1][1]
        known[5] = s_coeff_theory[1][2]
        known[6] = u_coeff_theory[0]
        known[7] = u_coeff_theory[1]
        known[8] = u_coeff_theory[2]
        
        return known


class Unknown(laplace_theory.TheoreticalValue):
    """ Values Related to Unknowns """

    def __init__(self, f_id, x, s, unknown, x_value):
        self.Theory = laplace_theory.TheoreticalValue(f_id, x, s, x_value)
        self.unknown = unknown
        
    def unknown_theory(self):
        unknown = self.unknown
        s_coeff_theory = self.Theory.s_coeff_theory()
        u_coeff_theory = self.Theory.u_coeff_theory()
        
        unknown_theory = np.ndarray((len(unknown),))
        unknown_theory[0] = s_coeff_theory[0][3]
        unknown_theory[1] = s_coeff_theory[0][4]
        unknown_theory[2] = s_coeff_theory[0][5]
        unknown_theory[3] = s_coeff_theory[1][3]
        unknown_theory[4] = s_coeff_theory[1][4]
        unknown_theory[5] = s_coeff_theory[1][5]
        unknown_theory[6] = u_coeff_theory[3]
        unknown_theory[7] = u_coeff_theory[4]
        unknown_theory[8] = u_coeff_theory[5]
        
        return unknown_theory
        
    def unknown_init(self, error_limit):
        unknown = self.unknown
        unknown_theory = self.unknown_theory()
    
        unknown_init = np.ndarray((len(unknown),))
        for i in range(len(unknown)):
            e = random.uniform(-error_limit, error_limit)
            unknown_init[i] = (1 + e/100)*unknown_theory[i]
        
        return unknown_init


class Taylor(Known):
    """ Taylor Series Expressions """
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init):
        self.Known = Known(f_id, x, s, unknown, x_value)
        self.x_value = x_value
        self.unknown_init = unknown_init
        
    def x_taylor_s(self, x, unknown):
        """ 2nd Order x_Taylor Series of s1 """
        x_value = self.x_value
        known = self.Known.known()

        dx = x - x_value
        
        x_taylor_s = np.ndarray((2,), 'object')
    
        x_taylor_s[0] = known[0] \
                        + known[1]*dx[0] \
                        + known[2]*dx[1] \
                        + unknown[0]*dx[0]**2/2 \
                        + unknown[1]*dx[0]*dx[1] \
                        + unknown[2]*dx[1]**2/2
               
        x_taylor_s[1] = known[3] \
                        + known[4]*dx[0] \
                        + known[5]*dx[1] \
                        + unknown[3]*dx[0]**2/2 \
                        + unknown[4]*dx[0]*dx[1] \
                        + unknown[5]*dx[1]**2/2
        
        return x_taylor_s
        
    def s_value(self):
        x_value = self.x_value
        unknown_init = self.unknown_init
        s_value = self.x_taylor_s(x_value, unknown_init)
        
        return s_value
        
    def s_taylor_u(self, s, unknown):
        """ 2nd Order x_Taylor Series of s1 """
        known = self.Known.known()
        s_value = self.s_value()
        
        ds = s - s_value
        
        s_taylor_u = known[6] \
                     + known[7]*ds[0] \
                     + known[8]*ds[1] \
                     + unknown[6]*ds[0]**2/2 \
                     + unknown[7]*ds[0]*ds[1] \
                     + unknown[8]*ds[1]**2/2
                     
        return s_taylor_u
    
    def x_taylor_u(self, x, unknown):
        """ Convolution of x_taylor_s & s_taylor_u """
        x_taylor_s = self.x_taylor_s(x, unknown)
        x_taylor_u = self.s_taylor_u(x_taylor_s, unknown)
        
        return x_taylor_u
    
 
class BoundaryConditions(Taylor):
    """ Boundary Conditions along Streamline """
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init, element_size):
        self.Taylor = Taylor(f_id, x, s, unknown, x_value, unknown_init)
        self.ProblemSettings = self.Taylor.Known.Theory.ProblemSettings
        self.x = x
        self.unknown = unknown
        self.x_value = x_value
        self.unknown_init = unknown_init
        self.element_size = element_size
        
    def s_boundary(self):
        x_value = self.x_value
        s_value = self.ProblemSettings.s(x_value)
        element_size = self.element_size
        
        s_boundary = np.ndarray((2, len(s_value)))
        s_boundary[0][0] = s_value[0] - element_size/2
        s_boundary[0][1] = s_value[1] 
        s_boundary[1][0] = s_value[0] + element_size/2
        s_boundary[1][1] = s_value[1] 
        
        return s_boundary
    
    def x_boundary(self):
        x = self.x
        x_value = self.x_value
        s_boundary = self.s_boundary()
        s = self.ProblemSettings.s(x)
        
        f = np.ndarray((2, len(x),), 'object')
        for i in range(2):
            for j in range(len(x)):
                f[i][j] = s[j] - s_boundary[i][j]

        x_boundary = np.ndarray((2, len(x),))
        for i in range(2):
            for j in range(len(x)):
                x_boundary[i][j] = syp.nsolve((f[i][0], f[i][1]), \
                                             (x[0], x[1]), \
                                             (x_value[0], x_value[1]) \
                                             )[j]
                
        return x_boundary
    
    def u_boundary(self):
        s_boundary = self.s_boundary()
        
        u_boundary = np.ndarray((2),)
        for i in range(2):
            u_boundary[i] = self.ProblemSettings.u(s_boundary[i])
        
        return u_boundary
    
    def boundary_conditions(self):
        unknown = self.unknown
        x_boundary = self.x_boundary()
        u_boundary = self.u_boundary()
        
        x_taylor_u = np.ndarray((2,), 'object')
        bc = np.ndarray((2,), 'object')
        for i in range(2):
            x_taylor_u[i] = self.Taylor.x_taylor_u(x_boundary[i], unknown)
            bc[i] = x_taylor_u[i] - u_boundary[i]
            
        return bc
        

class Derivative(Taylor):
    """ x_Taylor Series of Derivatives """
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init):
        self.Taylor = Taylor(f_id, x, s, unknown, x_value, unknown_init)
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
    
    def dds_ddx(self):
        x = self.x
        unknown = self.unknown
        x_taylor_s = self.Taylor.x_taylor_s(x, unknown)
        
        dds_ddx = np.ndarray((2, 2, 2), 'object')
        for i in range(2):
            dds_ddx[i][0][0] = diff(x_taylor_s[i], x[0], 2)
            dds_ddx[i][0][1] = diff(x_taylor_s[i], x[0], x[1])
            dds_ddx[i][1][0] = diff(x_taylor_s[i], x[1], x[0])
            dds_ddx[i][1][1] = diff(x_taylor_s[i], x[1], 2)
        
        return dds_ddx
        

class Metric(Derivative):
    """ x_Taylor Series of Metrics """
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init):
        self.Derivative = Derivative(f_id, x, s, unknown, x_value, unknown_init)
        self.x = x
    
    def supermetric(self):
        """ Superscript Metric g^ij """
        ds_dx = self.Derivative.ds_dx()
        
        ds1_dx = np.ndarray((2,), 'object')
        ds1_dx[0] = ds_dx[0][0]
        ds1_dx[1] = ds_dx[0][1]
        ds2_dx = np.ndarray((2,), 'object')
        ds2_dx[0] = ds_dx[1][0]
        ds2_dx[1] = ds_dx[1][1]
        
        supermetric = np.ndarray((2, 2,), 'object')
        supermetric[0][0] = dot(ds1_dx, ds1_dx)
        supermetric[0][1] = dot(ds1_dx, ds2_dx)
        supermetric[1][0] = dot(ds2_dx, ds1_dx)
        supermetric[1][1] = dot(ds2_dx, ds2_dx)
        
        return supermetric
    
    def dg_ds1(self):
        """ dg^ij/ds1 = dx1/ds1*dg^ij/dx1 + dx2/ds1*dg^ij/dx2 """
        dx1_ds1 = self.Derivative.dx_ds()[0][0]
        dx2_ds1 = self.Derivative.dx_ds()[1][0]
        supermetric = self.supermetric()
        x = self.x
        
        dg_dx1 = np.ndarray((2, 2), 'object')
        for i in range(2):
            for j in range(2):
                dg_dx1[i][j] = diff(supermetric[i][j], x[0]) 
                
        dg_dx2 = np.ndarray((2, 2), 'object')
        for i in range(2):
            for j in range(2):
                dg_dx2[i][j] = diff(supermetric[i][j], x[1])
                    
        dg_ds1 = np.ndarray((2, 2,), 'object')
        for i in range(2):
            for j in range(2):
                dg_ds1[i][j] = dx1_ds1*dg_dx1[i][j] \
                               + dx2_ds1*dg_dx2[i][j]
        
        return dg_ds1
        

class Laplacian(Metric):
    """ x_Taylor Series of Laplacian """
    
    def __init__(self, f_id, laplacian_id, x, s, unknown, x_value, unknown_init):
        self.Metric = Metric(f_id, x, s, unknown, x_value, unknown_init)
        self.Derivative = self.Metric.Derivative
        self.laplacian_id = laplacian_id
        
    def laplacian_u(self):
        """ g11*g22*u,11 + 1/2*(g22*g11,1 - g11*g22,1)*u,1 """
        laplacian_id = self.laplacian_id
        du_ds1 = self.Derivative.du_ds()[0]
        ddu_dds1 = self.Derivative.ddu_dds()[0][0]
        
        if laplacian_id == 'metric':
            g11 = self.Metric.supermetric()[0][0]
            g22 = self.Metric.supermetric()[1][1]
            dg11_ds1 = self.Metric.dg_ds1()[0][0]
            dg22_ds1 = self.Metric.dg_ds1()[1][1]
    
            laplacian_u = 2*g11*g22*ddu_dds1 \
                          + (g22*dg11_ds1 - g11*dg22_ds1)*du_ds1
            
        if laplacian_id == 'derivative':
            ds1_dx1 = self.Derivative.ds_dx()[0][0]
            ds1_dx2 = self.Derivative.ds_dx()[0][1]
            dds1_ddx1 = self.Derivative.dds_ddx()[0][0][0]
            dds1_ddx2 = self.Derivative.dds_ddx()[0][1][1]
    
            laplacian_u = ((ds1_dx1)**2 + (ds1_dx2)**2)*ddu_dds1 \
                          + (dds1_ddx1 + dds1_ddx2)*du_ds1    
        
        return laplacian_u


class GoverningEquations(Laplacian):
    """ Derive Governing Equations """
    
    def __init__(self, f_id, ge_id, x, s, unknown, x_value, unknown_init):
        self.Laplacian = Laplacian(f_id, ge_id, x, s, unknown, x_value, unknown_init)
        self.Metric = self.Laplacian.Metric
        self.Derivative = self.Laplacian.Metric.Derivative
        self.Taylor = self.Laplacian.Metric.Derivative.Taylor
        self.x =  x
        self.x_value = x_value

    def governing_equation_0(self):
        """ 1st Order x_Taylor Series of g_12 """
        du_ds2 = self.Derivative.du_ds()[1]
        x = self.x
        x_value = self.x_value
        
        coeff_du_ds2 = np.ndarray((6), 'object')
#        coeff_du_ds2[0] = du_ds2
        coeff_du_ds2[1] = diff(du_ds2, x[0])
        coeff_du_ds2[2] = diff(du_ds2, x[1])
#        coeff_du_ds2[3] = diff(du_ds2, x[0], 2)
#        coeff_du_ds2[4] = diff(du_ds2, x[0], x[1])
#        coeff_du_ds2[5] = diff(du_ds2, x[1], 2)
        
        for i in range(len(coeff_du_ds2)):
            coeff_du_ds2[i] = lambdify(x, coeff_du_ds2[i], 'numpy')
            coeff_du_ds2[i] = coeff_du_ds2[i](x_value[0], x_value[1])
            
        return coeff_du_ds2
        
    def governing_equation_1(self):
        """ 1st Order x_Taylor Series of g_12 """
        g12 = self.Metric.supermetric()[0][1]
        x = self.x
        x_value = self.x_value
        
        coeff_g12 = np.ndarray((6), 'object')
#        coeff_g12[0] = g12
        coeff_g12[1] = diff(g12, x[0])
        coeff_g12[2] = diff(g12, x[1])
        coeff_g12[3] = diff(g12, x[0], 2)
        coeff_g12[4] = diff(g12, x[0], x[1])
        coeff_g12[5] = diff(g12, x[1], 2)
        
        for i in range(len(coeff_g12)):
            coeff_g12[i] = lambdify(x, coeff_g12[i], 'numpy')
            coeff_g12[i] = coeff_g12[i](x_value[0], x_value[1])
            
        return coeff_g12
    
    def governing_equation_2(self):
        """ 1st Order x_Taylor Series of g_12 """
        laplacian_u = self.Laplacian.laplacian_u()
        x = self.x
        x_value = self.x_value
        
        coeff_laplacian_u = np.ndarray((6), 'object')
        coeff_laplacian_u[0] = laplacian_u
#        coeff_laplacian_u[1] = diff(laplacian_u, x[0])
#        coeff_laplacian_u[2] = diff(laplacian_u, x[1])
#        coeff_laplacian_u[3] = diff(laplacian_u, x[0], 2)
#        coeff_laplacian_u[4] = diff(laplacian_u, x[0], x[1])
#        coeff_laplacian_u[5] = diff(laplacian_u, x[1], 2)
        
        for i in range(len(coeff_laplacian_u)):
            coeff_laplacian_u[i] = lambdify(x, coeff_laplacian_u[i], 'numpy')
            coeff_laplacian_u[i] = coeff_laplacian_u[i](x_value[0], x_value[1])
            
        return coeff_laplacian_u


class Solve(BoundaryConditions, GoverningEquations):
    """ Solve BVP on Line Element by Newton's Method """
    
    def __init__(self, f_id, ge_id, solver_id, x, s, unknown, x_value, unknown_init, element_size, newton_tol):
        self.BC = BoundaryConditions(f_id, x, s, unknown, x_value, unknown_init, element_size)
        self.GE = GoverningEquations(f_id, ge_id, x, s, unknown, x_value, unknown_init)
        self.unknown = unknown
        self.unknown_init = unknown_init
        self.newton_tol = newton_tol
        self.solver_id = solver_id
    
    def f(self):
        unknown = self.unknown
        bc = self.BC.boundary_conditions()
        ge0 = self.GE.governing_equation_0()
        ge1 = self.GE.governing_equation_1()
        ge2 = self.GE.governing_equation_2()
        
        f = np.ndarray((len(unknown),), 'object')
        f[0] = bc[0]
        f[1] = bc[1]
        f[2] = ge0[1]
        f[3] = ge0[2]
        f[4] = ge1[1]
        f[5] = ge1[2]
        f[6] = ge1[3]
        f[7] = ge1[4]
        f[8] = ge2[0]
        
        return f
    
    def Jacobian_f(self, unknown_temp):
        unknown = self.unknown
        f = self.f()
        
        Jacobian_f = np.ndarray((len(f), len(unknown),), 'object')
        for i in range(len(f)):
            for j in range(len(unknown)):
                Jacobian_f[i][j] = diff(f[i], unknown[j])
                Jacobian_f[i][j] = lambdify(unknown, Jacobian_f[i][j], 'numpy')
                Jacobian_f[i][j] = Jacobian_f[i][j](unknown_temp[0],
                                                    unknown_temp[1],
                                                    unknown_temp[2],
                                                    unknown_temp[3],
                                                    unknown_temp[4],
                                                    unknown_temp[5],
                                                    unknown_temp[6],
                                                    unknown_temp[7],
                                                    unknown_temp[8],
                                                    )
        Jacobian_f = Jacobian_f.astype('double')
        
        return Jacobian_f
    
    def residual(self, unknown_temp):
        unknown = self.unknown
        f = self.f()
        
        residual = np.ndarray((len(f),), 'object')
        for i in range(len(f)):
            residual[i] = -f[i]
            for j in range(len(unknown)):
                residual[i] += diff(f[i], unknown[j])*unknown[j]
            residual[i] = lambdify(unknown, residual[i], 'numpy')
            residual[i] = residual[i](unknown_temp[0],
                                      unknown_temp[1],
                                      unknown_temp[2],
                                      unknown_temp[3],
                                      unknown_temp[4],
                                      unknown_temp[5],
                                      unknown_temp[6],
                                      unknown_temp[7],
                                      unknown_temp[8],
                                      )
        residual = residual.astype('double')
    
        return residual   
    
    def newton_error(self, unknown_temp):
        unknown = self.unknown
        f = self.f()
        
        newton_error = np.ndarray((len(f),), 'object')
        for i in range(len(f)):
            newton_error[i] = lambdify(unknown, f[i], 'numpy')
            newton_error[i] = newton_error[i](unknown_temp[0],
                                              unknown_temp[1],
                                              unknown_temp[2],
                                              unknown_temp[3],
                                              unknown_temp[4],
                                              unknown_temp[5],
                                              unknown_temp[6],
                                              unknown_temp[7],
                                              unknown_temp[8],
                                              )
        newton_error = norm(newton_error)
        
        return newton_error
    
    def solution(self):
        unknown_init = self.unknown_init
        unknown_temp = unknown_init
        newton_tol = self.newton_tol
        newton_error = self.newton_error(unknown_temp)
        solver_id = self.solver_id 
        
        while newton_error > newton_tol:
            Jacobian_f = self.Jacobian_f(unknown_temp)
            residual = self.residual(unknown_temp)
            
            if solver_id == 'np.solve':
                unknown_temp = np.linalg.solve(Jacobian_f, residual)
            if solver_id == 'np.lstsq':
                unknown_temp = np.linalg.lstsq(Jacobian_f, residual)[0]
            if solver_id == 'scp.spsolve':
                unknown_temp = scp.sparse.linalg.spsolve(Jacobian_f, residual)
            if solver_id == 'scp.bicg':
                unknown_temp = scp.sparse.linalg.bicg(Jacobian_f, residual)[0]
            if solver_id == 'scp.lsqr':
                unknown_temp = scp.sparse.linalg.lsqr(Jacobian_f, residual)[0]
                
            newton_error = self.newton_error(unknown_temp)
        
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
    
    unknown = np.ndarray((9,), 'object')
    unknown[0] = Symbol('s1_11', real = True)
    unknown[1] = Symbol('s1_12', real = True)
    unknown[2] = Symbol('s1_22', real = True)
    unknown[3] = Symbol('s2_11', real = True)
    unknown[4] = Symbol('s2_12', real = True)
    unknown[5] = Symbol('s2_22', real = True)
    unknown[6] = Symbol('u11', real = True)
    unknown[7] = Symbol('u12', real = True)
    unknown[8] = Symbol('u22', real = True)
    
    ################################
    f_id = 'z^2'
#    f_id = 'z^3'
#    f_id = 'z^4'
#    f_id = 'exp(z)'
    
#    laplacian_id = 'metric'
    laplacian_id = 'derivative'
    
    n = 1
    error_init_limit = 0.0
    element_size = 1.0e-1
    newton_tol = 1.0e-8
    
#    solver_id = 'np.solve'
    solver_id = 'np.lstsq'
#    solver_id = 'scp.spsolve'
#    solver_id = 'scp.bicg'
#    solver_id = 'scp.lsqr'
    ##############################
    
    print('')
    print('f(z) =', f_id)
    print('Laplacian =', laplacian_id)
    print('error_init_limit =', error_init_limit)
    print('element_size =', element_size)
    print('newton_tol =', newton_tol)
    print('solver =', solver_id)
    print('')
    
    x_target = np.ndarray((len(x),))
    
    x_target_array = np.ndarray((n, len(x),))
    
    unknown_theory_array = np.ndarray((n, len(unknown)))
    unknown_init_array = np.ndarray((n, len(unknown)))
    unknown_terminal_array = np.ndarray((n, len(unknown)))
    
    error_array = np.ndarray((n, 2))
    error_sum_array = np.ndarray((2))
    error_sum_array[0] = 0
    error_sum_array[1] = 0
    
    min_abs_eigvals_Jacobian_f_init_array = np.ndarray((n, 1))
    
    
    def relative_error(a, b):
        
        relative_error = round(norm(b - a)/norm(a), 4)*100
        
        return relative_error
    
    
    x_min = np.ndarray((2))
    x_min[0] = 0.0
    x_min[1] = 0.0
    
    x_max = np.ndarray((2))
    x_max[0] = 2.0
    x_max[1] = 2.0
    
    for i in range(n):
        x_target[0] = random.uniform(x_min[0], x_max[0])
        x_target[1] = random.uniform(x_min[1], x_max[1])
        
        ######################################################
        Unknown_call = Unknown(f_id, x, s, unknown, x_target)
        ######################################################
        unknown_theory = Unknown_call.unknown_theory()
        unknown_init = Unknown_call.unknown_init(error_init_limit)
    
        ###################################################################################################################
        Solve_call = Solve(f_id, laplacian_id, solver_id, x, s, unknown, x_target, unknown_init, element_size, newton_tol)
        ###################################################################################################################
        unknown_terminal = Solve_call.solution()
        f_init = Solve_call.f()
        Jacobian_f_init = Solve_call.Jacobian_f(unknown_init)
        eigvals_Jacobian_f_init = eigvals(Jacobian_f_init)
        abs_eigvals_Jacobian_f_init = abs(eigvals_Jacobian_f_init)
        min_abs_eigvals_Jacobian_f_init_array[i] = min(abs_eigvals_Jacobian_f_init)
        
        error_init = relative_error(unknown_theory, unknown_init)
        error_terminal = relative_error(unknown_theory, unknown_terminal)
        
        for j in range(len(x)):
            x_target_array[i][j] = x_target[j]

        for j in range(len(unknown)):
            unknown_theory_array[i][j] = unknown_theory[j]
            unknown_init_array[i][j] = unknown_init[j]
            unknown_terminal_array[i][j] = unknown_terminal[j]
            
        error_array[i][0] = error_init
        error_array[i][1] = error_terminal
        error_sum_array[0] += error_init
        error_sum_array[1] += error_terminal
    
    print('')
    print('x_target = ')
    print(x_target_array)
    print('')
    
    print('error_init(%) & error_terminal(%) = ')
    print(error_array)
    print('')
    
    print('error_init_sum(%) & error_terminal_sum(%) = ')
    print(error_sum_array)
    print('')
    
#    print('min_abs_eigvals_Jacobian_f_init = ')
#    print(min_abs_eigvals_Jacobian_f_init_array)
#    print('')
            
            
    t1 = time.time()
    
    print('Elapsed Time = ')
    print(round(t1 - t0), '(s)')
    print('')







        
    
    
    
    

