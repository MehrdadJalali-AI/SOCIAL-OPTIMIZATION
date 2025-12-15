"""Benchmark functions for optimization."""

import numpy as np
from typing import Tuple, Callable, Dict


class BenchmarkFunctions:
    """Collection of benchmark optimization functions."""
    
    def __init__(self):
        """Initialize benchmark functions dictionary."""
        self.functions: Dict[str, Tuple[Callable, list, float]] = {
            'Sphere': (self.sphere, [-100, 100], 0.0),
            'Schwefel_2_22': (self.schwefel_2_22, [-10, 10], 0.0),
            'Schwefel_1_2': (self.schwefel_1_2, [-100, 100], 0.0),
            'Schwefel_2_21': (self.schwefel_2_21, [-100, 100], 0.0),
            'Rosenbrock': (self.rosenbrock, [-30, 30], 0.0),
            'Step': (self.step, [-100, 100], 0.0),
            'Quartic': (self.quartic, [-1.28, 1.28], 0.0),
            'Schwefel_2_26': (self.schwefel_2_26, [-500, 500], 0.0),
            'Rastrigin': (self.rastrigin, [-5.12, 5.12], 0.0),
            'Ackley': (self.ackley, [-32, 32], 0.0),
            'Griewank': (self.griewank, [-600, 600], 0.0),
            'Penalized': (self.penalized, [-50, 50], 0.0),
            'Penalized2': (self.penalized2, [-50, 50], 0.0),
            'Foxholes': (self.foxholes, [-65.536, 65.536], 0.998),
            'Kowalik': (self.kowalik, [-5, 5], 0.0003075),
            'Camel-Back': (self.camel_back, [-5, 5], -1.0316),
            'Branin': (self.branin, [-5, 5], 0.398),
            'Goldstein-Price': (self.goldstein_price, [-2, 2], 3.0),
            'Hartman': (self.hartman3, [0, 1], -3.86),
            'Shekel1': (self.hartman6, [0, 1], -3.322),
            'Shekel2': (self.shekel5, [0, 10], -10.1532),
            'Shekel3': (self.shekel7, [0, 10], -10.4028),
            'Shekel4': (self.shekel10, [0, 10], -10.5363)
        }
    
    @staticmethod
    def sphere(x):
        return np.sum(x**2)
    
    @staticmethod
    def schwefel_2_22(x):
        return np.sum(np.abs(x)) + np.prod(np.abs(x))
    
    @staticmethod
    def schwefel_1_2(x):
        return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])
    
    @staticmethod
    def schwefel_2_21(x):
        return np.max(np.abs(x))
    
    @staticmethod
    def rosenbrock(x):
        return np.sum([100 * (x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1)])
    
    @staticmethod
    def step(x):
        return np.sum(np.floor(x+0.5)**2)
    
    @staticmethod
    def quartic(x):
        # Note: Original has noise, but for reproducibility we'll make it deterministic
        # If you need noise, add: + np.random.uniform(0, 1)
        return np.sum([(i+1)*xi**4 for i, xi in enumerate(x)])
    
    @staticmethod
    def schwefel_2_26(x):
        return 418.9829*len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def rastrigin(x):
        return 10 * len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    @staticmethod
    def ackley(x):
        return -20 * np.exp(-0.2*np.sqrt(np.sum(x**2)/len(x))) - \
               np.exp(np.sum(np.cos(2*np.pi*x))/len(x)) + 20 + np.e
    
    @staticmethod
    def griewank(x):
        return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1)))) + 1
    
    @staticmethod
    def penalized(x):
        term1 = (np.pi/len(x))*(10*np.sin(np.pi*(1+(x[0]+1)/4))**2 +
                               np.sum([((1+(x[i]+1)/4)-1)**2 * (1+10*np.sin(np.pi*(1+(x[i+1]+1)/4))**2)
                                       for i in range(len(x)-1)]) +
                               ((1+(x[-1]+1)/4)-1)**2)
        term2 = np.sum([100*(xi-10)**4 if xi>10 else (-10-xi)**4 if xi < -10 else 0 for xi in x])
        return term1 + term2
    
    @staticmethod
    def penalized2(x):
        term1 = 0.1*(np.sin(3*np.pi*x[0])**2 +
                    np.sum([(x[i]-1)**2 * (1+np.sin(3*np.pi*x[i+1])**2)
                            for i in range(len(x)-1)]) +
                    (x[-1]-1)**2 * (1+np.sin(2*np.pi*x[-1])**2))
        term2 = np.sum([0.1*(xi-5)**4 if xi>5 else (-5-xi)**4 if xi<-5 else 0 for xi in x])
        return term1 + term2
    
    @staticmethod
    def foxholes(x):
        x = x[:2]
        a = np.array([[4.0]*25, np.linspace(0, 12, 25)])
        denom = 1/500.0
        for j in range(25):
            sum_term = (x[0] - a[0, j])**6 + (x[1] - a[1, j])**6
            denom += 1.0 / (j + 1 + sum_term)
        return 1.0 / denom
    
    @staticmethod
    def kowalik(x):
        x = x[:4]
        a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844,
                      0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        b = np.array([4, 2, 1, 0.5, 0.25,
                      0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625])
        s = 0.0
        for i in range(11):
            s += (a[i] - (x[0]*(b[i]**2 + b[i]*x[1]) / (b[i]**2 + b[i]*x[2] + x[3]*x[2])))**2
        return s
    
    @staticmethod
    def camel_back(x):
        x = x[:2]
        return 4*x[0]**2 - 2.1*x[0]**4 + (1/3)*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4
    
    @staticmethod
    def branin(x):
        x = x[:2]
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * np.pi)
        return (x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*np.cos(x[0]) + s
    
    @staticmethod
    def goldstein_price(x):
        x = x[:2]
        term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[1] - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[1] + 48*x[0] - 36*x[0]*x[1] + 27*x[1]**2)
        return term1 * term2
    
    @staticmethod
    def hartman3(x):
        x = x[:3]
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]])
        P = 1e-4 * np.array([[3689, 1170, 2673],
                             [4699, 4387, 7470],
                             [1091, 8732, 5547],
                             [381, 5743, 8828]])
        outer = 0.0
        for i in range(4):
            inner = np.sum(A[i] * ((x - P[i])**2))
            outer += alpha[i] * np.exp(-inner)
        return -outer
    
    @staticmethod
    def hartman6(x):
        x = x[:6]
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])
        outer = 0.0
        for i in range(4):
            inner = np.sum(A[i] * ((x - P[i])**2))
            outer += alpha[i] * np.exp(-inner)
        return -outer
    
    @staticmethod
    def shekel5(x):
        x = x[:4]
        m = 5
        C = 0.1 * np.ones(m)
        A = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7]])
        sum_val = 0.0
        for i in range(m):
            diff = x - A[i]
            sum_val += 1.0 / (np.sum(diff**2) + C[i])
        return -sum_val
    
    @staticmethod
    def shekel7(x):
        x = x[:4]
        m = 7
        C = 0.1 * np.ones(m)
        A = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7],
                      [2, 9, 2, 9],
                      [5, 5, 3, 3]])
        sum_val = 0.0
        for i in range(m):
            diff = x - A[i]
            sum_val += 1.0 / (np.sum(diff**2) + C[i])
        return -sum_val
    
    @staticmethod
    def shekel10(x):
        x = x[:4]
        m = 10
        C = 0.1 * np.ones(m)
        A = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7],
                      [2, 9, 2, 9],
                      [5, 5, 3, 3],
                      [8, 1, 8, 1],
                      [6, 2, 6, 2],
                      [7, 3.6, 7, 3.6]])
        sum_val = 0.0
        for i in range(m):
            diff = x - A[i]
            sum_val += 1.0 / (np.sum(diff**2) + C[i])
        return -sum_val

