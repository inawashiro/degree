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
from sympy import nsolve

# For Measuring Computation Timea
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

        plt.xlabel('x1', labelpad = 8)
        plt.ylabel('x2', labelpad = 8)

        plt.savefig('./figure/' + f_id + '/u/theory.pdf')
        plt.savefig('./figure/' + f_id + '/u/theory.png')
        
        plt.pause(.01)
        
    def pcs_theory_plot(self):
        s_theory_plot = self.s_theory_plot()
        f_id = self.f_id
        x_plot = self.x_plot
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        interval1 = np.arange(-100, 100, 1.0)
        interval2 = np.arange(-100, 100, 1.0)
        
        s1 = plt.contour(x_plot[0], x_plot[1], s_theory_plot[0], interval1, 
                         colors = 'red')        
        s2 = plt.contour(x_plot[0], x_plot[1], s_theory_plot[1], interval2, 
                         colors = 'blue')
        
        plt.locator_params(axis = 'x', nbins = 5)
        plt.locator_params(axis = 'y', nbins = 5)
        
        plt.xlabel('x1', labelpad = 8)
        plt.ylabel('x2', labelpad = 8)
        
        labels = ['s1 = const.', 's2 = const.']
            
        s1.collections[0].set_label(labels[0])
        s2.collections[0].set_label(labels[1])

        plt.legend(bbox_to_anchor=(1.05, 1), loc = 2, borderaxespad = 0., 
                   frameon = False)
        
        plt.savefig('./figure/' + f_id + '/pcs/theory.pdf')
        plt.savefig('./figure/' + f_id + '/pcs/theory.png')
        
        plt.pause(.01)

        s_levels = []
        s_levels.append(s1.levels.tolist())
        s_levels.append(s2.levels.tolist())

        return s_levels
    
    def x_target_array(self):
        x = self.x
        s = self.ProblemSettings.s(x)
        s_levels = self.pcs_theory_plot()

        x_target_array = np.ndarray((len(s_levels[0]), len(s_levels[1]), 2))
        for i in range(len(s_levels[0])):
            for j in range(len(s_levels[1])):
                f1 = s[0] - s_levels[0][i]
                f2 = s[1] - s_levels[1][j]
                for k in range(len(x)):
                    x_target_array[i][j][k] = syp.nsolve((f1, f2), (x[0], x[1]), 
                                                         (1, 1))[k]
                    
        return x_target_array
        
        
class TerminalPlot(TheoryPlot):
    
    def __init__(self, x_min, x_max, x_target_array, 
                 unknown_terminal_error_array):
        self.TheoryPlot = TheoryPlot
        self.x_min = x_min
        self.x_max = x_max
        self.newton_tol = newton_tol
        self.x_target_array = x_target_array
        self.unknown_terminal_error_array = unknown_terminal_error_array
     
    def unknown_terminal_error_plot(self):
        s_theory_plot = self.TheoryPlot.s_theory_plot()
        x_min = self.x_min
        x_max = self.x_max
        x_target_array = self.x_target_array
        unknown_terminal_error_array = self.unknown_terminal_error_array
        
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        interval1 = np.arange(-100, 100, 1.0)
        interval2 = np.arange(-100, 100, 1.0)
        
        plt.contour(x_plot[0], x_plot[1], s_theory_plot[0], interval1, 
                    colors = 'gray', linestyles = 'dotted')        
        plt.contour(x_plot[0], x_plot[1], s_theory_plot[1], interval2, 
                    colors = 'gray', linestyles = 'dotted')
        
        error_plot = plt.scatter(x_target_array[:, :, 0], 
                                 x_target_array[:, :, 1], 
                                 s = 30,
                                 vmin = 0,
                                 vmax = 100,
                                 c = unknown_terminal_error_array, 
                                 cmap = cm.seismic)
        
        plt.xlim(x_min[0], x_max[0])
        plt.ylim(x_min[1], x_max[1])
        
        plt.locator_params(axis = 'x', nbins = 5)
        plt.locator_params(axis = 'y', nbins = 5)
        
        plt.xlabel('x1', labelpad = 8)
        plt.ylabel('x2', labelpad = 8)
        
        plt.colorbar(error_plot)
        
        plt.savefig('./figure/' + f_id + '/unknown/terminal_error.pdf')
        plt.savefig('./figure/' + f_id + '/unknown/terminal_error.png')
        
        plt.pause(.01)
        
    def unknown_terminal_error_hist_plot(self):
        unknown_terminal_error_array = self.unknown_terminal_error_array
        
        unknown_terminal_error_array = np.ravel(unknown_terminal_error_array)
        
        bins = range(0, 210, 5)
        weights = np.ones_like(unknown_terminal_error_array) / len(unknown_terminal_error_array)
        
        plt.hist(unknown_terminal_error_array, bins = bins, weights = weights)
        
        plt.axvline(unknown_terminal_error_array.mean(), color = 'k', 
                    linestyle = 'dashed', linewidth = 1)

        _, max_ = plt.ylim()
        plt.text(unknown_terminal_error_array.mean() + 10, max_*0.9, 
                 'Mean: {:.1f}'.format(unknown_terminal_error_array.mean()))
        
        plt.locator_params(axis = 'x', nbins = 11)
        plt.locator_params(axis = 'y', nbins = 5)
        
        plt.xlabel('Error (%)', labelpad = 8)
        plt.ylabel('Density', labelpad = 8)
        
        plt.savefig('./figure/' + f_id + '/unknown/terminal_error_histogram.pdf')
        plt.savefig('./figure/' + f_id + '/unknown/terminal_error_histogram.png')
        
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
    
    ###########################################################################
#    f_id = 'z^2'
#    element_size = 1.0e-1
#    newton_tol = 1.0e-7
#    x_min = np.ndarray((2))
#    x_min[0] = 0.0
#    x_min[1] = 0.0
#    x_max = np.ndarray((2))
#    x_max[0] = 2.0
#    x_max[1] = 2.0
    
#    f_id = 'z^3'
#    element_size = 1.0e-2
#    newton_tol = 1.0e-7
#    x_min = np.ndarray((2))
#    x_min[0] = 1.2
#    x_min[1] = 1.2
#    x_max = np.ndarray((2))
#    x_max[0] = 2.0
#    x_max[1] = 2.0

    f_id = 'exp(kz)'
    element_size = 8.0e-2
    newton_tol = 1.0e-9
    x_min = np.ndarray((2))
    x_min[0] = 0.0
    x_min[1] = 0.0
    x_max = np.ndarray((2))
    x_max[0] = 2.0
    x_max[1] = 2.0
   
    unknown_init_error_limit = 400.0
    taylor_order = 2
    ###########################################################################
    
    print('')
    print('u = Re{',f_id,'}')
    print('')
    print('unknown_init_error_limit =', unknown_init_error_limit)
    print('element_size =', element_size)
    print('newton_tol =', newton_tol)
    print('')

    
    def relative_error(a, b):
        
        relative_error = round(np.linalg.norm(b - a)/np.linalg.norm(a), 4)*100
        
        return relative_error


    x_sidelength = np.ndarray((2))
    x_sidelength[0] = x_max[0] - x_min[0]
    x_sidelength[1] = x_max[1] - x_min[1]
    
    x_plot = np.meshgrid(np.arange(x_min[0], x_max[0], (x_sidelength[0])/500), 
                         np.arange(x_min[1], x_max[1], (x_sidelength[1])/500))
    
    ############################################
    TheoryPlot = TheoryPlot(f_id, x, s, x_plot)
    ############################################
    x_target_array = TheoryPlot.x_target_array()
    
    number_of_points = len(x_target_array)*len(x_target_array[0])
    
    unknown_terminal_error_array = np.ndarray((len(x_target_array), 
                                               len(x_target_array[0])))
    
    u_init_array = np.ndarray((len(x_target_array), 
                               len(x_target_array[0])))
    
    u_terminal_array = np.ndarray((len(x_target_array), 
                                   len(x_target_array[0])))
    
    unknown_init_error_mean = 0
    unknown_terminal_error_mean = 0
    
    for i in range(len(x_target_array)):
        for j in range(len(x_target_array[i])):
            x_target = x_target_array[i][j]
    
            unknown_init_error = np.random.uniform(-unknown_init_error_limit, 
                                                   unknown_init_error_limit)
            ###################################################################
            Unknown_call = laplace_experiment.Unknown(f_id, x, s, unknown, 
                                                      x_target, 
                                                      unknown_init_error)
            ###################################################################
            unknown_theory = Unknown_call.unknown_theory()
            unknown_init = Unknown_call.unknown_init()
            
            ###################################################################
            Solve_call = laplace_experiment.Solve(f_id, x, s, unknown,
                                                  x_target, unknown_init_error,
                                                  element_size, taylor_order)
            ###################################################################
            unknown_terminal = Solve_call.solution(newton_tol)
            
            unknown_terminal_error = relative_error(unknown_theory, unknown_terminal)
            unknown_terminal_error_array[i][j] = unknown_terminal_error
            
            unknown_init_error_mean += abs(unknown_init_error)/number_of_points
            unknown_terminal_error_mean += unknown_terminal_error/number_of_points
            
    print('unknown_init_error_mean (%) = ')
    print(unknown_init_error_mean)
    print('') 
    
    print('unknown_terminal_error_mean (%) = ')
    print(unknown_terminal_error_mean)
    print('') 
    
    ###########################################################################
    TerminalPlot = TerminalPlot(x_min, x_max, x_target_array,
                                unknown_terminal_error_array)
    ###########################################################################

    print('unknown_terminal_error Distribution')
    TerminalPlot.unknown_terminal_error_plot()
    print('')      
    
    print('unknown_terminal_error Histogram')
    TerminalPlot.unknown_terminal_error_hist_plot()
    print('') 
    
    
    t1 = time.time()
    
    print('Elapsed Time = ', round(t1 - t0), '(s)')        
        
     
    
    
    
    