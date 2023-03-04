"""Volume 2: Simplex

Nathan Schill
Section 2
Thurs. Mar. 9, 2023
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """

        # Raise error if any entry in b is < 0
        if any(b < 0):
            raise ValueError('the system is infeasible at the origin')

        # Create dictionary
        self.dictionary = self._generatedictionary(c, A, b)

        # print(self.dictionary)
        

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        
        # Get shape
        m, _ = A.shape
        
        # Create cbar and Abar
        cbar = np.pad(c, (0,m))
        Abar = np.hstack((A, np.eye(m)))

        # Create vertical stacks
        cAstack = np.vstack((cbar, -Abar))
        bpad = np.expand_dims(np.pad(b, (1,0)), 1)
        
        # Return D
        return np.hstack((bpad, cAstack))


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """

        # Get indices of negative coefficients in first row
        # (add 1 since searching after first column)
        negative_indices = np.where(self.dictionary[0, 1:] < 0)[0] + 1
        
        return negative_indices[0]


    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """

        # Get indices of negative coefficients
        negative_indices = np.where(self.dictionary[1:, index] < 0)[0] + 1
        
        # Get ratios of negative entries
        all_ratios = -self.dictionary[negative_indices, 0] / \
            self.dictionary[negative_indices, index]

        # Get row of lowest ratio
        return negative_indices[np.argmin(all_ratios)]


    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        # Get pivot col
        col = self._pivot_col()

        # Check whether all entries in pivot column are non-negative.
        # If so, problem is unbounded and has no solution.
        if np.all(self.dictionary[col] >= 0):
            raise ValueError('problem is unbounded and has no solution')

        # Get pivot row and pivot entry, and divide pivot row by -pivot
        row = self._pivot_row(col)
        pivot = self.dictionary[row, col]
        self.dictionary[row, :] /= -pivot
        
        # Get indices of rows other than pivot row
        other_row_indices = np.arange(len(self.dictionary)) != row

        # For each non-pivot row, add to it the pivot row
        # times the non-pivot row's pivot col entry
        self.dictionary[other_row_indices, :] += self.dictionary[row, :] * \
            np.expand_dims(self.dictionary[other_row_indices, col], 1)
        

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        
        # While any entry in top row (besides first) is negative, continue solving
        while np.any(self.dictionary[0, 1:] < 0):
            self.pivot()

        # Get variable indices of non-zero coefficients in first row of dictionary
        dependent_indices = np.where(~np.isclose(self.dictionary[0, 1:], 0))[0] + 1
        
        # Get variable indices of zero coefficients in first row of dictionary
        independent_indices = np.where(np.isclose(self.dictionary[0, 1:], 0))[0] + 1

        # Get row indices of the -1 coefficients, and populate independent dict
        var_indices = np.argmin(self.dictionary[:, independent_indices], axis=0)
        independent_dict = \
            {i-1:self.dictionary[row, 0] for i, row in zip(independent_indices, var_indices)}

        # Dependent var dictionary gets entry from top row corresponding to each var
        return self.dictionary[0,0], \
            independent_dict, \
            {i-1:0 for i in dependent_indices}

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """

    # Load and unpack data
    data = np.load(filename)
    A = data['A']
    p = data['p']
    m = data['m']
    d = data['d']
    
    # Number of products
    n = len(p)

    # Gives objective to minimize
    c = -p

    # Get constraints
    b = np.concatenate((m, d))

    # Construct matrix
    B = np.vstack((A, np.eye(len(d))))

    # Solve
    ss = SimplexSolver(c, B, b)
    soln = ss.solve()

    # Get the number of products from whichever dictionary it's in
    num_products = np.zeros(n)
    for i in range(n):
        if i in soln[1]:
            num_products[i] = soln[1][i]
        else:
            num_products[i] = soln[2][i]
    
    return num_products