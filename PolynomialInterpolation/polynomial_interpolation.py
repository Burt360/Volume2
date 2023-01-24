# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Nathan Schill
Section 2
Thurs. Jan. 26, 2023
"""

import numpy as np
import matplotlib.pyplot as plt


# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    
    def Lj(j, xint, yint, points):
        '''Evaluate L_j at points.'''

        # Compute denominator
        denom = np.product(xint[j-1] - np.delete(xint, j-1))
        
        # Tranpose points to column. From each point subtract
        # each point in xint (except jth), yielding a row for each point in points.
        # Take product along each row. Divide each entry by denom.
        return np.product(points[:,np.newaxis] - np.delete(xint, j-1), axis=1)/denom

    # Stack the n Ljs (each an array of m values where Lj was evaluated),
    # yielding an nxm array. Sum down columns to get get m y-values of interpolation.
    return np.sum(np.array([yint[j-1] * Lj(j, xint, yint, points) for j in range(1, len(xint)+1)]), axis=0)


# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        
        # Store interpolating points
        self.xint, self.yint = xint, yint

        ### Use code given on pp. 106-7 in the lab PDF

        # Number of interpolating points
        n = len(self.xint)
        
        # Array for storing barycentric weights
        self.w = np.ones(n)

        # Calculate the capacity of the interval
        C = (np.max(self.xint) - np.min(self.xint)) / 4

        shuffle = np.random.permutation(n-1)
        for j in range(n):
            temp = (self.xint[j] - np.delete(self.xint, j)) / C
            # Randomize order of product
            temp = temp[shuffle]
            self.w[j] /= np.product(temp)

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        
        # Nudge any points found in xint to avoid division by zero
        points[np.isin(points, self.xint)] += 0.1
        # return [points - self.xint[j] for j in range(len(self.xint))]

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    raise NotImplementedError("Problem 7 Incomplete")
