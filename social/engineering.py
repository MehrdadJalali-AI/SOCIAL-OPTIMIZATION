"""Engineering design problems with constraints."""

import numpy as np
from typing import Tuple, List, Dict
from .variables import VariableSpec, repair_variable


class EngineeringProblems:
    """Collection of constrained engineering optimization problems."""
    
    def __init__(self):
        """Initialize engineering problems."""
        self.problems = {
            'Pressure_Vessel': self.pressure_vessel,
            'Welded_Beam': self.welded_beam,
            'Tension_Compression_Spring': self.tension_compression_spring,
            'Speed_Reducer': self.speed_reducer,
            'Cantilever_Beam': self.cantilever_beam,
            'Gear_Train': self.gear_train
        }
        
        # Variable specifications
        self.var_specs = {
            'Pressure_Vessel': [
                VariableSpec("real", [0.0625, 6.1875]),  # x1 (discrete in practice)
                VariableSpec("real", [0.0625, 6.1875]),  # x2 (discrete in practice)
                VariableSpec("real", [10.0, 200.0]),    # x3
                VariableSpec("real", [10.0, 200.0])     # x4
            ],
            'Welded_Beam': [
                VariableSpec("real", [0.1, 2.0]),   # h
                VariableSpec("real", [0.1, 10.0]),  # l
                VariableSpec("real", [0.1, 10.0]),  # t
                VariableSpec("real", [0.1, 2.0])    # b
            ],
            'Tension_Compression_Spring': [
                VariableSpec("real", [0.05, 2.0]),   # d
                VariableSpec("real", [0.25, 1.3]),  # D
                VariableSpec("int", [2, 15])         # N (integer)
            ],
            'Speed_Reducer': [
                VariableSpec("real", [2.6, 3.6]),   # x1
                VariableSpec("real", [0.7, 0.8]),   # x2
                VariableSpec("int", [17, 28]),      # x3
                VariableSpec("real", [7.3, 8.3]),   # x4
                VariableSpec("real", [7.3, 8.3]),   # x5
                VariableSpec("real", [2.9, 3.9]),  # x6
                VariableSpec("real", [5.0, 5.5])   # x7
            ],
            'Cantilever_Beam': [
                VariableSpec("real", [1.0, 5.0]) for _ in range(5)  # x1-x5
            ],
            'Gear_Train': [
                VariableSpec("int", [12, 60]) for _ in range(4)  # All integers
            ]
        }
    
    def pressure_vessel(self, x: np.ndarray) -> Tuple[float, List[float]]:
        """
        Pressure Vessel Design Problem.
        
        Minimize: f(x) = 0.6224*x1*x3*x4 + 1.7781*x2*x3^2 + 3.1661*x1^2*x4 + 19.84*x1^2*x3
        
        Constraints:
        g1: -x1 + 0.0193*x3 <= 0
        g2: -x2 + 0.00954*x3 <= 0
        g3: -π*x3^2*x4 - (4/3)*π*x3^3 + 1296000 <= 0
        g4: x4 - 240 <= 0
        """
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        
        # Objective
        f = 0.6224*x1*x3*x4 + 1.7781*x2*x3**2 + 3.1661*x1**2*x4 + 19.84*x1**2*x3
        
        # Constraints
        g1 = -x1 + 0.0193*x3
        g2 = -x2 + 0.00954*x3
        g3 = -np.pi*x3**2*x4 - (4/3)*np.pi*x3**3 + 1296000
        g4 = x4 - 240
        
        constraints = [g1, g2, g3, g4]
        return f, constraints
    
    def welded_beam(self, x: np.ndarray) -> Tuple[float, List[float]]:
        """
        Welded Beam Design Problem.
        
        Minimize: f(x) = 1.10471*x1^2*x2 + 0.04811*x3*x4*(14+x2)
        
        Constraints:
        g1: τ(x) - τ_max <= 0
        g2: σ(x) - σ_max <= 0
        g3: x1 - x4 <= 0
        g4: 0.10471*x1^2 + 0.04811*x3*x4*(14+x2) - 5 <= 0
        g5: 0.125 - x1 <= 0
        g6: δ(x) - δ_max <= 0
        g7: P - Pc(x) <= 0
        """
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        
        # Constants
        P = 6000
        L = 14
        E = 30e6
        G = 12e6
        tau_max = 13600
        sigma_max = 30000
        delta_max = 0.25
        
        # Objective
        f = 1.10471*x1**2*x2 + 0.04811*x3*x4*(14+x2)
        
        # Stress calculations
        M = P * (L + x2/2)
        R = np.sqrt(x2**2/4 + ((x1+x3)/2)**2)
        J = 2*(x1*x2*np.sqrt(2)*(x2**2/12 + ((x1+x3)/2)**2))
        tau1 = P / (np.sqrt(2)*x1*x2)
        tau2 = M*R / J
        tau = np.sqrt(tau1**2 + 2*tau1*tau2*x2/(2*R) + tau2**2)
        
        sigma = 6*P*L / (x4*x3**2)
        delta = 4*P*L**3 / (E*x3**3*x4)
        Pc = (4.013*E*np.sqrt(x3**2*x4**6/36)) / (L**2) * (1 - x3/(2*L)*np.sqrt(E/(4*G)))
        
        # Constraints
        g1 = tau - tau_max
        g2 = sigma - sigma_max
        g3 = x1 - x4
        g4 = 0.10471*x1**2 + 0.04811*x3*x4*(14+x2) - 5
        g5 = 0.125 - x1
        g6 = delta - delta_max
        g7 = P - Pc
        
        constraints = [g1, g2, g3, g4, g5, g6, g7]
        return f, constraints
    
    def tension_compression_spring(self, x: np.ndarray) -> Tuple[float, List[float]]:
        """
        Tension/Compression Spring Design Problem.
        
        Minimize: f(x) = (x3+2)*x2*x1^2
        
        Constraints:
        g1: 1 - (x2^3*x3)/(71785*x1^4) <= 0
        g2: (4*x2^2 - x1*x2)/(12566*(x2*x1^3 - x1^4)) + 1/(5108*x1^2) - 1 <= 0
        g3: 1 - (140.45*x1)/(x2^2*x3) <= 0
        g4: (x1+x2)/1.5 - 1 <= 0
        """
        x1, x2, x3 = x[0], x[1], int(x[2])  # x3 is integer
        
        # Objective
        f = (x3+2)*x2*x1**2
        
        # Constraints
        g1 = 1 - (x2**3*x3)/(71785*x1**4)
        g2 = (4*x2**2 - x1*x2)/(12566*(x2*x1**3 - x1**4)) + 1/(5108*x1**2) - 1
        g3 = 1 - (140.45*x1)/(x2**2*x3)
        g4 = (x1+x2)/1.5 - 1
        
        constraints = [g1, g2, g3, g4]
        return f, constraints
    
    def speed_reducer(self, x: np.ndarray) -> Tuple[float, List[float]]:
        """
        Speed Reducer Design Problem.
        
        Minimize: f(x) = 0.7854*x1*x2^2*(3.3333*x3^2+14.9334*x3-43.0934) - 
                         1.508*x1*(x6^2+x7^2) + 7.4777*(x6^3+x7^3) + 
                         0.7854*(x4*x6^2+x5*x7^2)
        
        Constraints: 11 constraints (simplified version)
        """
        x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], int(x[2]), x[3], x[4], x[5], x[6]
        
        # Objective
        f = (0.7854*x1*x2**2*(3.3333*x3**2+14.9334*x3-43.0934) -
             1.508*x1*(x6**2+x7**2) + 7.4777*(x6**3+x7**3) +
             0.7854*(x4*x6**2+x5*x7**2))
        
        # Constraints (simplified set)
        g1 = 27/(x1*x2**2*x3) - 1
        g2 = 397.5/(x1*x2**2*x3**2) - 1
        g3 = 1.93*x4**3/(x2*x3*x6**4) - 1
        g4 = 1.93*x5**3/(x2*x3*x7**4) - 1
        g5 = np.sqrt((745*x4/(x2*x3))**2 + 16.9e6) / (0.1*x6**3) - 1100
        g6 = np.sqrt((745*x5/(x2*x3))**2 + 157.5e6) / (0.1*x7**3) - 850
        g7 = x2*x3/40 - 1
        g8 = 5*x2/x1 - 1
        g9 = x1/(12*x2) - 1
        g10 = (1.5*x6+1.9)/x4 - 1
        g11 = (1.1*x7+1.9)/x5 - 1
        
        constraints = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]
        return f, constraints
    
    def cantilever_beam(self, x: np.ndarray) -> Tuple[float, List[float]]:
        """
        Cantilever Beam Design Problem.
        
        Minimize: f(x) = 0.0624 * sum(x_i)
        
        Constraints:
        g: 61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3 - 1 <= 0
        """
        # Objective
        f = 0.0624 * np.sum(x)
        
        # Constraint
        g = 61/x[0]**3 + 37/x[1]**3 + 19/x[2]**3 + 7/x[3]**3 + 1/x[4]**3 - 1
        
        constraints = [g]
        return f, constraints
    
    def gear_train(self, x: np.ndarray) -> Tuple[float, List[float]]:
        """
        Gear Train Design Problem (discrete/integer).
        
        Minimize: f(x) = ((1/6.931) - (x3*x2)/(x1*x4))^2
        
        No constraints (all variables are integers in [12, 60])
        """
        x1, x2, x3, x4 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
        
        # Objective
        f = ((1/6.931) - (x3*x2)/(x1*x4))**2
        
        # No constraints
        constraints = []
        return f, constraints
    
    def evaluate_constrained(self, problem_name: str, x: np.ndarray,
                           repair_mode: str = "round") -> Dict:
        """
        Evaluate constrained problem with feasibility checking.
        
        Args:
            problem_name: Name of problem
            x: Design vector
            repair_mode: Repair mode for variables
            
        Returns:
            Dictionary with objective, constraints, feasibility info
        """
        if problem_name not in self.problems:
            raise ValueError(f"Unknown problem: {problem_name}")
        
        # Repair variables if needed
        if problem_name in self.var_specs:
            x_repaired = repair_variable(x, self.var_specs[problem_name], repair_mode)
        else:
            x_repaired = x
        
        # Evaluate
        obj, constraints = self.problems[problem_name](x_repaired)
        
        # Check feasibility
        violations = [max(0, g) for g in constraints]
        is_feasible = all(g <= 0 for g in constraints)
        total_violation = sum(violations)
        max_violation = max(violations) if violations else 0.0
        
        return {
            'objective': obj,
            'constraints': constraints,
            'violations': violations,
            'is_feasible': is_feasible,
            'total_violation': total_violation,
            'max_violation': max_violation,
            'x_repaired': x_repaired
        }
    
    def deb_feasibility_rule(self, obj1: float, violations1: List[float],
                            obj2: float, violations2: List[float]) -> int:
        """
        Deb's feasibility rule for comparing solutions.
        
        Returns:
            -1 if solution 1 is better, 1 if solution 2 is better, 0 if equal
        """
        feasible1 = all(v <= 0 for v in violations1)
        feasible2 = all(v <= 0 for v in violations2)
        
        if feasible1 and feasible2:
            # Both feasible: compare objectives
            return -1 if obj1 < obj2 else (1 if obj2 < obj1 else 0)
        elif feasible1:
            # Solution 1 feasible, solution 2 not
            return -1
        elif feasible2:
            # Solution 2 feasible, solution 1 not
            return 1
        else:
            # Both infeasible: compare violations
            total_viol1 = sum(max(0, v) for v in violations1)
            total_viol2 = sum(max(0, v) for v in violations2)
            return -1 if total_viol1 < total_viol2 else (1 if total_viol2 < total_viol1 else 0)

