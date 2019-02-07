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
from scipy.sparse.linalg import spsolve, lsqr, lsmr



class Known(laplace_theory.TheoryValue):
    """ Known Values """
    
    def __init__(self, f_id, x, s, unknown, x_value):
        self.Theory = laplace_theory.TheoryValue(f_id, x, s, x_value)
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


class Unknown(laplace_theory.TheoryValue):
    """ Unknown Values """

    def __init__(self, f_id, x, s, unknown, x_value, unknown_init_error):
        self.Theory = laplace_theory.TheoryValue(f_id, x, s, x_value)
        self.unknown = unknown
        self.unknown_init_error = unknown_init_error
        
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
        
    def unknown_init(self):
        unknown = self.unknown
        unknown_init_error = self.unknown_init_error
        unknown_theory = self.unknown_theory()
    
        unknown_init = np.ndarray((len(unknown),))
        for i in range(len(unknown)):
            unknown_init[i] = (1 + unknown_init_error/100)*unknown_theory[i]
        
        return unknown_init


class Taylor(Known, Unknown):
    """ Taylor Series Expansion """
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init_error):
        self.Known = Known(f_id, x, s, unknown, x_value)
        self.Unknown = Unknown(f_id, x, s, unknown, x_value, unknown_init_error)
        self.x_value = x_value
        
    def x_taylor_s(self, x, unknown):
        """ 2nd Order x_Taylor Series Expansion of s """
        x_value = self.x_value
        known = self.Known.known()

        dx = x - x_value
        
        x_taylor_s = np.ndarray((2,), 'object')
    
        x_taylor_s[0] = known[0] \
                        + known[1]*dx[0] \
                        + known[2]*dx[1] \
                        + unknown[0]*dx[0]**2 \
                        + unknown[1]*dx[0]*dx[1] \
                        + unknown[2]*dx[1]**2
               
        x_taylor_s[1] = known[3] \
                        + known[4]*dx[0] \
                        + known[5]*dx[1] \
                        + unknown[3]*dx[0]**2 \
                        + unknown[4]*dx[0]*dx[1] \
                        + unknown[5]*dx[1]**2
        
        return x_taylor_s
        
    def s_value(self):
        x_value = self.x_value
        unknown_init = self.Unknown.unknown_init()
        s_value = self.x_taylor_s(x_value, unknown_init)
        
        return s_value
        
    def s_taylor_u(self, s, unknown):
        """ 2nd Order s_Taylor Series Expansion of u """
        known = self.Known.known()
        s_value = self.s_value()
        
        ds = s - s_value
        
        s_taylor_u = known[6] \
                     + known[7]*ds[0] \
                     + known[8]*ds[1] \
                     + unknown[6]*ds[0]**2 \
                     + unknown[7]*ds[0]*ds[1] \
                     + unknown[8]*ds[1]**2
                     
        return s_taylor_u
    
    def x_polynomial_u(self, x, unknown):
        """ x_polynomial Series of u """
        """ Convolution of x_taylor_s & s_taylor_u """
        x_taylor_s = self.x_taylor_s(x, unknown)
        x_polynomial_u = self.s_taylor_u(x_taylor_s, unknown)
        
        return x_polynomial_u
    
 
class BoundaryConditions(Taylor):
    """ Boundary Conditions along Each Line Element """
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init_error, 
                 element_size):
        self.Taylor = Taylor(f_id, x, s, unknown, x_value, unknown_init_error)
        self.ProblemSettings = self.Taylor.Known.Theory.ProblemSettings
        self.x = x
        self.unknown = unknown
        self.x_value = x_value
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
        
        x_polynomial_u = np.ndarray((2,), 'object')
        bc = np.ndarray((2,), 'object')
        for i in range(2):
            x_polynomial_u[i] = self.Taylor.x_polynomial_u(x_boundary[i], unknown)
            bc[i] = x_polynomial_u[i] - u_boundary[i]
            
        return bc
        

class Derivative(Taylor):
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init_error):
        self.Taylor = Taylor(f_id, x, s, unknown, x_value, unknown_init_error)
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
            du_ds[i] = syp.diff(s_taylor_u, s[i])
        
        for i in range(2):
            du_ds[i] = syp.lambdify(s, du_ds[i], 'numpy')
            du_ds[i] = du_ds[i](x_taylor_s[0], x_taylor_s[1])
        
        return du_ds
                    
    def ddu_dds(self):
        x = self.x
        s = self.s
        unknown = self.unknown
        s_taylor_u = self.Taylor.s_taylor_u(s, unknown)
        x_taylor_s = self.Taylor.x_taylor_s(x, unknown)
        
        ddu_dds = np.ndarray((2, 2), 'object')
        ddu_dds[0][0] = syp.diff(s_taylor_u, s[0], 2)
        ddu_dds[0][1] = syp.diff(s_taylor_u, s[0], s[1])
        ddu_dds[1][0] = syp.diff(s_taylor_u, s[1], s[0])
        ddu_dds[1][1] = syp.diff(s_taylor_u, s[1], 2)
        
        for i in range(2):
            for j in range(2):
                ddu_dds[i][j] = syp.lambdify(s, ddu_dds[i][j], 'numpy')
                ddu_dds[i][j] = ddu_dds[i][j](x_taylor_s[0], x_taylor_s[1])       
        
        return ddu_dds
       
    def ds_dx(self):
        x = self.x
        unknown = self.unknown
        x_taylor_s = self.Taylor.x_taylor_s(x, unknown)
        
        ds_dx = np.ndarray((2, 2,), 'object')
        ds_dx[0][0] = syp.diff(x_taylor_s[0], x[0])
        ds_dx[0][1] = syp.diff(x_taylor_s[0], x[1])
        ds_dx[1][0] = syp.diff(x_taylor_s[1], x[0])
        ds_dx[1][1] = syp.diff(x_taylor_s[1], x[1])
                
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
            dds_ddx[i][0][0] = syp.diff(x_taylor_s[i], x[0], 2)
            dds_ddx[i][0][1] = syp.diff(x_taylor_s[i], x[0], x[1])
            dds_ddx[i][1][0] = syp.diff(x_taylor_s[i], x[1], x[0])
            dds_ddx[i][1][1] = syp.diff(x_taylor_s[i], x[1], 2)
        
        return dds_ddx
        

class Metric(Derivative):
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init_error):
        self.Derivative = Derivative(f_id, x, s, unknown, x_value, 
                                     unknown_init_error)
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
        supermetric[0][0] = np.dot(ds1_dx, ds1_dx)
        supermetric[0][1] = np.dot(ds1_dx, ds2_dx)
        supermetric[1][0] = np.dot(ds2_dx, ds1_dx)
        supermetric[1][1] = np.dot(ds2_dx, ds2_dx)
        
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
                dg_dx1[i][j] = syp.diff(supermetric[i][j], x[0]) 
                
        dg_dx2 = np.ndarray((2, 2), 'object')
        for i in range(2):
            for j in range(2):
                dg_dx2[i][j] = syp.diff(supermetric[i][j], x[1])
                    
        dg_ds1 = np.ndarray((2, 2,), 'object')
        for i in range(2):
            for j in range(2):
                dg_ds1[i][j] = dx1_ds1*dg_dx1[i][j] \
                               + dx2_ds1*dg_dx2[i][j]
        
        return dg_ds1
        

class Laplacian(Metric):
    """ x_Taylor Series of Laplacian """
    
    def __init__(self, f_id, x, s, unknown, x_value, 
                 unknown_init_error):
        self.Metric = Metric(f_id, x, s, unknown, x_value, unknown_init_error)
        self.Derivative = self.Metric.Derivative
        
    def laplacian_u(self):
        """ g11*g22*u,11 + 1/2*(g22*g11,1 - g11*g22,1)*u,1 """
        du_ds1 = self.Derivative.du_ds()[0]
        ddu_dds1 = self.Derivative.ddu_dds()[0][0]
        
        ds1_dx1 = self.Derivative.ds_dx()[0][0]
        ds1_dx2 = self.Derivative.ds_dx()[0][1]
        dds1_ddx1 = self.Derivative.dds_ddx()[0][0][0]
        dds1_ddx2 = self.Derivative.dds_ddx()[0][1][1]

        laplacian_u = ((ds1_dx1)**2 + (ds1_dx2)**2)*ddu_dds1 \
                      + (dds1_ddx1 + dds1_ddx2)*du_ds1    
        
        return laplacian_u


class GoverningEquations(Laplacian):
    """ Derive Governing Equations """
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init_error, 
                 taylor_order):
        self.Laplacian = Laplacian(f_id, x, s, unknown, x_value, unknown_init_error)
        self.Metric = self.Laplacian.Metric
        self.Derivative = self.Laplacian.Metric.Derivative
        self.Taylor = self.Laplacian.Metric.Derivative.Taylor
        self.taylor_order = taylor_order
        self.x = x
        self.s = s
        self.unknown = unknown
        self.x_value = x_value

    def governing_equation_0(self):
        """ Coefficients of du/ds2 """
        du_ds2 = self.Derivative.du_ds()[1]
        N = self.taylor_order
        x = self.x
        x_value = self.x_value
        
        len_coeff_du_ds2 = int(1/2*(N + 1)*(N + 2))
        
        coeff_du_ds2 = np.ndarray((len_coeff_du_ds2), 'object')
        coeff_du_ds2[0] = du_ds2
        coeff_du_ds2[1] = syp.diff(du_ds2, x[0])
        coeff_du_ds2[2] = syp.diff(du_ds2, x[1])
        coeff_du_ds2[3] = syp.diff(du_ds2, x[0], 2)
        coeff_du_ds2[4] = syp.diff(du_ds2, x[0], x[1])
        coeff_du_ds2[5] = syp.diff(du_ds2, x[1], 2)
        
        for i in range(len(coeff_du_ds2)):
            coeff_du_ds2[i] = syp.lambdify(x, coeff_du_ds2[i], 'numpy')
            coeff_du_ds2[i] = coeff_du_ds2[i](x_value[0], x_value[1])
            
        return coeff_du_ds2
        
    def governing_equation_1(self):
        """ Coefficients of g_12 """
        g12 = self.Metric.supermetric()[0][1]
        N = self.taylor_order
        x = self.x
        x_value = self.x_value
        
        len_coeff_g12 = int(1/2*(N + 1)*(N + 2))
        
        coeff_g12 = np.ndarray((len_coeff_g12), 'object')
        coeff_g12[0] = g12
        coeff_g12[1] = syp.diff(g12, x[0])
        coeff_g12[2] = syp.diff(g12, x[1])
        coeff_g12[3] = syp.diff(g12, x[0], 2)
        coeff_g12[4] = syp.diff(g12, x[0], x[1])
        coeff_g12[5] = syp.diff(g12, x[1], 2)
        
        for i in range(len(coeff_g12)):
            coeff_g12[i] = syp.lambdify(x, coeff_g12[i], 'numpy')
            coeff_g12[i] = coeff_g12[i](x_value[0], x_value[1])
            
        return coeff_g12
    
    def governing_equation_2(self):
        """ Coefficients of Δu """
        laplacian_u = self.Laplacian.laplacian_u()
        N = self.taylor_order
        x = self.x
        x_value = self.x_value
        
        len_coeff_laplacian_u = int(1/2*(N + 1)*(N + 2))
        
        coeff_laplacian_u = np.ndarray((len_coeff_laplacian_u), 'object')
        coeff_laplacian_u[0] = laplacian_u
        coeff_laplacian_u[1] = syp.diff(laplacian_u, x[0])
        coeff_laplacian_u[2] = syp.diff(laplacian_u, x[1])
        coeff_laplacian_u[3] = syp.diff(laplacian_u, x[0], 2)
        coeff_laplacian_u[4] = syp.diff(laplacian_u, x[0], x[1])
        coeff_laplacian_u[5] = syp.diff(laplacian_u, x[1], 2)
        
        for i in range(len(coeff_laplacian_u)):
            coeff_laplacian_u[i] = syp.lambdify(x, coeff_laplacian_u[i], 'numpy')
            coeff_laplacian_u[i] = coeff_laplacian_u[i](x_value[0], x_value[1])
            
        return coeff_laplacian_u


class Solve(BoundaryConditions, GoverningEquations):
    """ Solve BVP of Each Line Element by Non-linear Least Square Algorithm """
    
    def __init__(self, f_id, x, s, unknown, x_value, unknown_init_error, 
                 element_size, taylor_order):
        self.BC = BoundaryConditions(f_id, x, s, unknown, x_value, 
                                     unknown_init_error, element_size)
        self.GE = GoverningEquations(f_id, x, s, unknown, x_value, 
                                     unknown_init_error, taylor_order)
        self.Unknown = self.BC.Taylor.Unknown
        self.taylor_order = taylor_order
        self.unknown = unknown
    
    def f(self):
        bc = self.BC.boundary_conditions()
        ge0 = self.GE.governing_equation_0()
        ge1 = self.GE.governing_equation_1()
        ge2 = self.GE.governing_equation_2()
        taylor_order = self.taylor_order
        
        len_f = int(3/2*(taylor_order + 1)*(taylor_order + 2) + 2)
        
        f = np.ndarray((len_f), 'object')
        f[0] = bc[0]
        f[1] = bc[1]
        f[2] = ge0[0]
        f[3] = ge0[1]
        f[4] = ge0[2]
        f[5] = ge0[3]
        f[6] = ge0[4]
        f[7] = ge0[5]
        f[8] = ge1[0]
        f[9] = ge1[1]
        f[10] = ge1[2]
        f[11] = ge1[3]
        f[12] = ge1[4]
        f[13] = ge1[5]
        f[14] = ge2[0]
        f[15] = ge2[1]
        f[16] = ge2[2]
        f[17] = ge2[3]
        f[18] = ge2[4]
        f[19] = ge2[5]
        
        return f
    
    def jacobian_f(self, unknown_temp):
        unknown = self.unknown
        f = self.f()
        
        jacobian_f = np.ndarray((len(f), len(unknown),), 'object')
        for i in range(len(f)):
            for j in range(len(unknown)):
                jacobian_f[i][j] = syp.diff(f[i], unknown[j])
                jacobian_f[i][j] = syp.lambdify(unknown, jacobian_f[i][j], 'numpy')
                jacobian_f[i][j] = jacobian_f[i][j](unknown_temp[0],
                                                    unknown_temp[1],
                                                    unknown_temp[2],
                                                    unknown_temp[3],
                                                    unknown_temp[4],
                                                    unknown_temp[5],
                                                    unknown_temp[6],
                                                    unknown_temp[7],
                                                    unknown_temp[8],
                                                    )
        jacobian_f = jacobian_f.astype('double')
        
        return jacobian_f
    
    def residual(self, unknown_temp):
        unknown = self.unknown
        f = self.f()
        
        residual = np.ndarray((len(f),), 'object')
        for i in range(len(f)):
            residual[i] = f[i]
            residual[i] = syp.lambdify(unknown, residual[i], 'numpy')
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
    
    def solution(self, newton_tol):
        unknown_temp = self.Unknown.unknown_init()
        jacobian_f = self.jacobian_f(unknown_temp)
        residual = self.residual(unknown_temp)
            
        while np.linalg.norm(residual) > newton_tol:
            increment = np.linalg.lstsq(jacobian_f, -residual)[0]            
            unknown_temp += increment
            jacobian_f = self.jacobian_f(unknown_temp)
            residual = self.residual(unknown_temp)
        solution = unknown_temp
        
        return solution






