# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Nathan Schill
Section 2
Thurs. Mar. 23, 2023
"""

import cvxpy as cp
import numpy as np


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    # Init x and objective
    x = cp.Variable(3, nonneg=True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    # Init constraint matrices
    G = np.array([[1, 2,  0],
                  [0, 1, -4]])
    h = np.array([3, 1])

    P = np.array([2, 10, 3])
    q = 12

    # Init problem
    constraints = [G@x <= h, P@x >= q]
    problem = cp.Problem(objective, constraints)

    # Solve
    soln = problem.solve()

    return soln, x.value
    

# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    # Init x and objective
    x = cp.Variable(4, nonneg=True)
    objective = cp.Minimize(cp.norm(x, 1))

    # Init problem
    constraints = [A@x == b]
    problem = cp.Problem(objective, constraints)

    # Solve
    soln = problem.solve()

    return soln, x.value


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    # Init p and objective (cost along each route)
    p = cp.Variable(6, nonneg=True)
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c.T @ p)

    # Supply constraints (one cap for each supply center)
    supply_mat = np.array([[1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1]])
    supply = np.array([7, 2, 4])

    # Demand requirements
    demand_mat = np.array([[1, 0, 1, 0, 1, 0],
                           [0, 1, 0, 1, 0, 1]])
    demand = np.array([5, 8])

    # Init problem
    constraints = [supply_mat @ p <= supply, demand_mat @ p == demand]
    problem = cp.Problem(objective, constraints)

    # Solve
    soln = problem.solve()

    return soln, p.value


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    # Init x and objective
    x = cp.Variable(3)
    Q = np.array([[3,2,1],
                  [2,4,2],
                  [1,2,3]])
    r = np.array([3,0,1])
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + r.T@x)

    # Init problem
    problem = cp.Problem(objective)

    # Solve
    soln = problem.solve()

    return soln, x.value


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """

    m, n = A.shape
    
    # Init x and objective
    x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.norm(A@x - b, 2))

    # Init problem
    # |x| = 1 is just the sum since x has non-negative entries
    constraints = [cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)

    # Solve
    soln = problem.solve()

    return soln, x.value


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    
    # Load data
    data = np.load('food.npy', allow_pickle=True).T

    # Init x and objective
    x = cp.Variable(18, nonneg=True)
    p = data[0]
    objective = cp.Minimize(p.T @ x)

    # Less than constraints (2-4)
    less_than_constraints = data[2:5]
    less_than_vector = [2000, 65, 50]
    
    # Greater than constraints (5-7)
    greater_than_constraints = data[5:8]
    greater_than_vector = [1000, 25, 46]

    # Init problem
    constraints = [less_than_constraints @ x <= less_than_vector,
                   greater_than_constraints @ x >= greater_than_vector]
    problem = cp.Problem(objective, constraints)

    # Solve
    soln = problem.solve()

    return soln, x.value