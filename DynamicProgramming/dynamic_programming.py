# dynamic_programming.py
"""Volume 2: Dynamic Programming.
Nathan Schill
Section 2
Thurs. Apr. 13, 2023
"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    
    # Expected values
    values = np.zeros(N)

    # Iterate backwards through array
    for i in range(N-2, -1, -1):
        # Use equation in lab PDF to compute expected value
        current = max((i+1)/(i+2) * values[i+1] + 1/N, values[i+1])
        
        # Optimal stopping point found; don't continue computing values
        if current == values[i+1]:
            break
        else:
            values[i] = current

    # Get argmax and return it and its expected value
    argmax = np.argmax(values)
    return values[argmax], argmax+1


# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    
    N = np.arange(3, M+1)
    
    # Calculate stopping percentages and plot
    optimal_stopping_percentages = np.array([calc_stopping(n)[1]/n for n in N])
    plt.plot(N, optimal_stopping_percentages, label='Optimal stopping percentage')

    # Plot properties
    plt.xlabel('$N$')
    plt.ylabel('%')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return optimal_stopping_percentages[-1]


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    
    # Possible amounts of cake
    w = np.arange(N+1)/N
    
    # Consumption matrix
    C = np.zeros((N+1, N+1))

    # Modify each column
    for col in range(N+1):
        # Mask the diagonal and above
        C[col+1:, col] = u(w[col+1:] - w[col])
    
    return C


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.
    The function u should accept Numpy arrays.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """

    # Possible amounts of cake
    w = np.arange(N+1)/N
    
    # Value matrix and policy matrix
    A = np.zeros((N+1, T+1))
    P = np.zeros_like(A)

    # Last column of A; last column of P
    A[:, T] = u(w)
    P[:, T] = w

    # Get consumption and lower triangle indices
    C = get_consumption(N, u=u)
    ind = np.tril_indices_from(C, 0)

    # Iterate backwards through time/the columns of A
    for t in range(T-1, -1, -1):
        # Calculate CVt
        CVt = np.zeros_like(C)
        CVt[ind] = (C + B*A[:, t+1])[ind]

        # Populate P
        policy_ind = np.argmax(CVt, axis=1)
        P[:,t] = w - w[policy_ind]

        # Populate A
        A[:,t] = np.take_along_axis(CVt, policy_ind[:,np.newaxis], axis=1).squeeze()
    
    return A, P


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    
    # Get policy
    _, P = eat_cake(T, N, B, u)

    # Possible amounts of cake
    w = np.arange(N+1)/N

    # Init policy vector
    policy = np.zeros(T+1)

    # Step from bottom left of P
    i = N
    for j in range(T+1):
        # Get amount to eat
        amount = P[i,j]

        # Get number of steps
        step = round(amount*N)
        policy[j] = amount

        # Step
        i -= step
        j += 1
    
    return policy