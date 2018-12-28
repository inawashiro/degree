# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:43:47 2018

@author: inawashiro
"""
import laplace_theory, laplace_experiment

# For Visualization

from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['contour.negative_linestyle'] = 'solid'

# For Numerical Computation 
import numpy as np

# For Symbolic Notation
import sympy as syp

# For Scientific Notation
from decimal import getcontext, Decimal

# For Measuring Computation Time
import time



class TheoryPlot(laplace_theory.ProblemSettings):
    
    def __init__(self, f_id, x, s, x_plot):
        self.ProblemSettings = laplace_theory.ProblemSettings(f_id)
        self.f_id = f_id
        self.x = x
        self.s = s
        self.x_plot = x_plot
    
    def s_theory_plot(self):
        x = self.x
        s_theory = self.ProblemSettings.s(x)
        x_plot = self.x_plot
        
        s_theory_plot = np.ndarray((len(s_theory),), 'object')
        s_theory_plot = syp.lambdify(x, s_theory, 'numpy')
        s_theory_plot = s_theory_plot(x_plot[0], x_plot[1])
        
        return s_theory_plot
    
    def u_theory_plot(self):
        f_id = self.f_id
        x_plot = self.x_plot
        s_theory_plot = self.s_theory_plot()
        u_theory_plot = self.ProblemSettings.u(s_theory_plot)
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.plot_wireframe(x_plot[0], x_plot[1], u_theory_plot, linewidth = 0.2)
    
        plt.locator_params(axis = 'x', nbins = 5)
        plt.locator_params(axis = 'y', nbins = 5)
        plt.locator_params(axis = 'z', nbins = 5)

        plt.xlabel('x1', labelpad = 12)
        plt.ylabel('x2', labelpad = 12)

        plt.savefig('./graph/' + f_id + '/u/theory.pdf')
        plt.savefig('./graph/' + f_id + '/u/theory.png')
        
        plt.pause(.01)
        
    def pcs_theory_plot(self):
        f_id = self.f_id
        x_plot = self.x_plot
        s_theory_plot = self.s_theory_plot()
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        interval1 = np.arange(-100, 100, 1.0)
        interval2 = np.arange(-100, 100, 1.0)
        
        s1 = plt.contour(x_plot[0], x_plot[1], s_theory_plot[0], interval1, colors = 'red')        
        s2 = plt.contour(x_plot[0], x_plot[1], s_theory_plot[1], interval2, colors = 'blue')
        
        plt.locator_params(axis = 'x', nbins = 5)
        plt.locator_params(axis = 'y', nbins = 5)
        
        plt.xlabel('x1', labelpad = 12)
        plt.ylabel('x2', labelpad = 12)
        
        labels = ['s1 = const.', 's2 = const.']
            
        s1.collections[0].set_label(labels[0])
        s2.collections[0].set_label(labels[1])

        plt.legend(bbox_to_anchor=(1.05, 1), loc = 2, borderaxespad = 0.)
        
        plt.savefig('./graph/' + f_id + '/pcs/theory.pdf')
        plt.savefig('./graph/' + f_id + '/pcs/theory.png')
        
        plt.pause(.01)


class InitPlot():
    
    def __init__(self, x_min, x_max, unknown_init_error, x_target_array, u_init_array):
        self.x_min = x_min
        self.x_max = x_max
        self.unknown_init_error = unknown_init_error
        self.x_target_array = x_target_array
        self.u_init_array = u_init_array
    
    def u_init_plot(self):
        x_min = self.x_min
        x_max = self.x_max
        x_target_array = self.x_target_array
        u_init_array = self.u_init_array
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.scatter(x_target_array[:, :, 0], x_target_array[:, :, 1], u_init_array[:, :])
        
        plt.xlim(x_min[0], x_max[0])
        plt.ylim(x_min[1], x_max[1])
        
        plt.locator_params(axis = 'x', nbins = 5)
        plt.locator_params(axis = 'y', nbins = 5)
        plt.locator_params(axis = 'z', nbins = 5)
        
        plt.xlabel('x1', labelpad = 12)
        plt.ylabel('x2', labelpad = 12)
        
        plt.savefig('./graph/' + f_id + '/u/init.pdf')
        plt.savefig('./graph/' + f_id + '/u/init.png')
        
        plt.pause(.01)

        
class TerminalPlot():
    
    def __init__(self, x_min, x_max, x_target_array, u_terminal_array, unknown_terminal_error_array):
        self.x_min = x_min
        self.x_max = x_max
        self.newton_tol = newton_tol
        self.x_target_array = x_target_array
        self.u_terminal_array = u_terminal_array
        self.unknown_terminal_error_array = unknown_terminal_error_array
    
    def u_terminal_plot(self):
        x_min = self.x_min
        x_max = self.x_max
        x_target_array = self.x_target_array
        u_terminal_array = self.u_terminal_array
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        ax.scatter(x_target_array[:, :, 0], x_target_array[:, :, 1], u_terminal_array[:, :])
        
        plt.xlim(x_min[0], x_max[0])
        plt.ylim(x_min[1], x_max[1])
        
        plt.locator_params(axis = 'x', nbins = 5)
        plt.locator_params(axis = 'y', nbins = 5)
        plt.locator_params(axis = 'z', nbins = 5)
        
        plt.xlabel('x1', labelpad = 12)
        plt.ylabel('x2', labelpad = 12)
        
        plt.savefig('./graph/' + f_id + '/u/terminal.pdf')
        plt.savefig('./graph/' + f_id + '/u/terminal.png')
        
        plt.pause(.01)
    
    def unknown_terminal_error_plot(self):
        x_min = self.x_min
        x_max = self.x_max
#        newton_tol = str(self.newton_tol)
        x_target_array = self.x_target_array
        unknown_terminal_error_array = self.unknown_terminal_error_array
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        error_plot = plt.scatter(x_target_array[:, :, 0], 
                                 x_target_array[:, :, 1], 
                                 s = 12,
                                 vmin = 0,
                                 vmax = 100,
                                 c = unknown_terminal_error_array, 
                                 cmap = cm.seismic)
        
        plt.xlim(x_min[0], x_max[0])
        plt.ylim(x_min[1], x_max[1])
        
        plt.locator_params(axis = 'x', nbins = 5)
        plt.locator_params(axis = 'y', nbins = 5)
        
        plt.xlabel('x1', labelpad = 12)
        plt.ylabel('x2', labelpad = 12)
        
        plt.colorbar(error_plot)
        
        plt.savefig('./graph/' + f_id + '/unknown/terminal_error.pdf')
        plt.savefig('./graph/' + f_id + '/unknown/terminal_error.png')
        
        plt.pause(.01)
        
        
        
if __name__ == '__main__':
    
    t0 = time.time()
    
    
    x = np.ndarray((2,), 'object')
    x[0] = syp.Symbol('x1', real = True)
    x[1] = syp.Symbol('x2', real = True)
    
    s = np.ndarray((2,), 'object')
    s[0] = syp.Symbol('s1', real = True)
    s[1] = syp.Symbol('s2', real = True)
    
    unknown = np.ndarray((9,), 'object')
    unknown[0] = syp.Symbol('s1_11', real = True)
    unknown[1] = syp.Symbol('s1_12', real = True)
    unknown[2] = syp.Symbol('s1_22', real = True)
    unknown[3] = syp.Symbol('s2_11', real = True)
    unknown[4] = syp.Symbol('s2_12', real = True)
    unknown[5] = syp.Symbol('s2_22', real = True)
    unknown[6] = syp.Symbol('u11', real = True)
    unknown[7] = syp.Symbol('u12', real = True)
    unknown[8] = syp.Symbol('u22', real = True)
    
    ################################
    f_id = 'z^2'
#    f_id = 'z^3'
#    f_id = 'exp(z)'
    
    x_min = np.ndarray((2))
    x_min[0] = 0.0
    x_min[1] = 0.0
    
    x_max = np.ndarray((2))
    x_max[0] = 2.0
    x_max[1] = 2.0
    
    x_sidelength = np.ndarray((2))
    x_sidelength[0] = x_max[0] - x_min[0]
    x_sidelength[1] = x_max[1] - x_min[1]
    
    x_plot = np.meshgrid(np.arange(x_min[0], x_max[0], (x_sidelength[0])/500), 
                         np.arange(x_min[1], x_max[1], (x_sidelength[1])/500))
    
#    formulation_id = 'metric'
    formulation_id = 'derivative'
    
    highest_order = 2
    number_of_partitions = 10
    unknown_init_error = 200.0
    element_size = 1.0e-1
    newton_tol = 1.0e-8
    
#    solver_id = 'np.solve'
    solver_id = 'np.lstsq'
#    solver_id = 'scp.spsolve'
#    solver_id = 'scp.bicg'
#    solver_id = 'scp.lsqr'
#    solver_id = 'scp.lsmr'
    ##############################
    print('')
    print('u = Re{',f_id,'}')
    print('')
    print('# of points = ', number_of_partitions - 1, 'x', number_of_partitions - 1)
    print('unknown_init_error =', unknown_init_error)
    print('element_size =', element_size)
    print('newton_tol =', newton_tol)
    print('solver =', solver_id)
    print('')

    
    def relative_error(a, b):
        
        relative_error = round(np.linalg.norm(b - a)/np.linalg.norm(a), 4)*100
        
        return relative_error


    x_target = np.ndarray((len(x),))
    
    x_target_array = np.ndarray((number_of_partitions - 1, 
                                 number_of_partitions - 1, 
                                 len(x),))
    
    unknown_terminal_error_array = np.ndarray((number_of_partitions - 1, 
                                               number_of_partitions - 1))
    
    u_init_array = np.ndarray((number_of_partitions - 1, 
                               number_of_partitions - 1,))
    
    u_terminal_array = np.ndarray((number_of_partitions - 1, 
                                   number_of_partitions - 1,))
    
    unknown_init_error_mean = 0
    unknown_terminal_error_mean = 0
    
    for i in range(number_of_partitions - 1):
        for j in range(number_of_partitions - 1):
            x_target[0] = (x_max[0] - x_min[0])*(i + 1)/number_of_partitions
            x_target[1] = (x_max[1] - x_min[1])*(j + 1)/number_of_partitions    
            for k in range(len(x)):
                x_target_array[i][j][k] = x_target[k]
    
            ######################################################
            Unknown_call = laplace_experiment.Unknown(f_id, x, s, unknown, x_target)
            ######################################################
            unknown_theory = Unknown_call.unknown_theory()
            unknown_init = Unknown_call.unknown_init(unknown_init_error)
            unknown_init_error = relative_error(unknown_theory, unknown_init)
            
            #############################################################################################################
            Solve_call = laplace_experiment.Solve(f_id, formulation_id, highest_order, x, s, unknown, x_target, unknown_init, element_size)
            #############################################################################################################
            unknown_terminal = Solve_call.solution(newton_tol, solver_id)
            
            unknown_terminal_error = relative_error(unknown_theory, unknown_terminal)
            unknown_terminal_error_array[i][j] = unknown_terminal_error
            
            unknown_init_error_mean += unknown_init_error/((number_of_partitions - 1)**2)
            unknown_terminal_error_mean += unknown_terminal_error/((number_of_partitions - 1)**2)
            
            #####################################################################################
            Taylor_call = laplace_experiment.Taylor(f_id, x, s, unknown, x_target, unknown_init)
            #####################################################################################
            s_init = Taylor_call.x_taylor_s(x_target, unknown_init)
            u_init = Taylor_call.s_taylor_u(s_init, unknown_init)
            u_init_array[i][j] = u_init
            
            s_terminal = Taylor_call.x_taylor_s(x_target, unknown_terminal)
            u_terminal = Taylor_call.s_taylor_u(s_terminal, unknown_terminal)
            u_terminal_array[i][j] = u_terminal
            
    print('unknown_init_error_mean(%) = ')
    print(unknown_init_error_mean)
    print('') 
    
    print('unknown_terminal_error_mean(%) = ')
    print(unknown_terminal_error_mean)
    print('') 
    
    ############################################
    TheoryPlot = TheoryPlot(f_id, x, s, x_plot)
    ############################################
    print('u_theory')
    TheoryPlot.u_theory_plot()
    print('')
    
    print('pcs_theory')
    TheoryPlot.pcs_theory_plot()
    print('')    
    
    ####################################################################################
    InitPlot = InitPlot(x_min, x_max, unknown_init_error, x_target_array, u_init_array)
    ####################################################################################
    print('u_init')
    InitPlot.u_init_plot()
    print('')
    
    ##########################################################################################################
    TerminalPlot = TerminalPlot(x_min, x_max, x_target_array, u_terminal_array, unknown_terminal_error_array)
    ##########################################################################################################
    print('u_terminal')
    TerminalPlot.u_terminal_plot()
    print('')
    
    print('unknown_terminal_error Distribution')
    TerminalPlot.unknown_terminal_error_plot()
    print('')      
    
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')        
        
     
    
    
    
    
    
    
    
    
    
    
    