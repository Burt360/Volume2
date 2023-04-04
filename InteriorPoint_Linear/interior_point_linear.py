# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
Nathan Schill
Section 2
Thurs. Apr. 6, 2023
"""

import numpy as np
from scipy import linalg as la
# from scipy.stats import linregress
from scipy import stats
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """

    m, n = A.shape

    # Centering parameter
    SIGMA = 1/10
    
    def F(x, lam, mu):
        '''Define F as in the lab PDF'''
        return np.concatenate((A.T @ lam + mu - c, A @ x - b, np.diag(mu) @ x))

    # The top two rows of the block form of DF(x, lam, mu)
    DF_top = np.vstack((
                np.hstack((
                    np.zeros((n,n)), A.T, np.eye(n,n)
                )),
                np.hstack((
                    A, np.zeros((m,m)), np.zeros((m,n))
                ))
            ))
    
    def search_dir(x, lam, mu, nu):
        '''Solve equation (16.2)'''
        
        e = np.ones(n)
        
        # DF(x, lam, mu)
        DF = np.vstack((DF_top,
                        np.hstack((
                            np.diag(mu), np.zeros((n,m)), np.diag(x)
                        ))
                      ))

        # Solve the equation
        lu_piv = la.lu_factor(DF)
        dir = la.lu_solve(lu_piv, -F(x, lam, mu) + np.concatenate(
            (np.zeros(n), np.zeros(m), SIGMA * nu * e)
            )
        )
        
        # grad_x, grad_lam, grad_mu
        return dir[:n], dir[n:n+m], dir[n+m:]
    
    def step_size(x, mu, grad_x, grad_mu):
        # Check if any mu has any negative entries
        if any(mu < 0):
            grad_mu_lt_zero = grad_mu < 0
            a_max = np.min(-mu[grad_mu_lt_zero]/grad_mu[grad_mu_lt_zero])
        # If all entries of mu non-negative
        else:
            a_max = 1
        
        # Check if any x has any negative entries
        if any(x < 0):
            grad_x_lt_zero = grad_x < 0
            d_max = np.min(-x[grad_x_lt_zero]/grad_x[grad_x_lt_zero])
        # If all entries of x non-negative
        else:
            d_max = 1
        
        return min(1, 0.95*a_max), min(1, 0.95*d_max)

    # Get starting point
    x, lam, mu = starting_point(A, b, c)

    for _ in range(niter):
        # Duality measure
        nu = x@mu / n

        # Check if reached tolerance
        if nu < tol:
            break
        
        # Get search direction
        grad_x, grad_lam, grad_mu = search_dir(x, lam, mu, nu)
        
        # Get step size
        a, d = step_size(x, mu, grad_x, grad_mu)

        # Step
        x += d*grad_x
        lam += a*grad_lam
        mu += a*grad_mu
    
    return x, c@x


def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    
    # Load data
    data = np.loadtxt(filename)

    # Init problem
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    # Init more problem
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    # Solve
    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]
    
    # Plot LAD
    domain = np.linspace(0, 10, 200)
    plt.scatter(data[:, 1], data[:, 0], label='Data')
    plt.plot(domain, domain*beta + b, label='LAD')

    # Plot least-squares solution
    slope, intercept = stats.linregress(data[:, 1], data[:, 0])[:2]
    plt.plot(domain, domain*slope + intercept, label='Least-squares')

    # Plot
    plt.legend()
    plt.show()