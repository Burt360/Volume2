# numpy_intro.py
"""Python Essentials: Intro to NumPy.
Nathan Schill
Section 3
Thurs. Sept. 15, 2022
"""

import numpy as np

def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""

    A = np.array( [ [3, -1, 4],
                    [1, 5, -9] ] )
    B = np.array( [ [2, 6, -5, 3],
                    [5, -8, 9, 7],
                    [9, -3, -2, -3] ] )
    
    # Two equivalent ways to return the matrix product...
    return A@B
    return np.dot(A, B)


def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    
    A = np.array( [ [3, 1, 4],
                    [1, 5, 9],
                    [-5, 3, 1] ] )

    # Return the expression specfied in the doc string.
    return -1*(A@A@A) + 9*(A@A) - 15*A


def prob3():
    """Define the matrices A and B as arrays using the functions presented in
    this section of the manual (not np.array()). Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    
    # An upper triangular 7x7 matrix of ones.
    A = np.triu(np.ones((7, 7)))

    # 5-6 = -1 on the lower triangle, and 5 above the lower triangle.
    B = np.full((7, 7), 5) - np.tril(np.full((7, 7), 6))

    return A@B@A


def prob4(A):
    """Make a copy of 'A' and use fancy indexing to set all negative entries of
    the copy to 0. Return the resulting array.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    
    B = A.copy()
    # Set any negative entry in B to 0.
    B[B < 0] = 0
    return B


def prob5():
    """Define the matrices A, B, and C as arrays. Use NumPy's stacking functions
    to create and return the block matrix:
                                | 0 A^T I |
                                | A  0  0 |
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    
    # Populate A with integers from 0 to 5, shape it into a 3x2 array, and transpose it.
    A = np.arange(6).reshape((3, 2)).T

    # B is a lower triangular matrix of 3s.
    B = np.tril(np.full((3, 3), 3))

    # C has a diagonal of -2s, and zeros elsewhere.
    C = np.diag([-2 for _ in range(3)])
    
    # Each vstack is a column of matrices, and these are concatenated together horizontally.
    return  np.hstack(( np.vstack((np.zeros((3, 3)), A, B)),
                        np.vstack((A.T, np.zeros((2, 2)), np.zeros((3, 2)))),
                        np.vstack((np.eye(3), np.zeros((2, 3)), C))))

def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.
    Use array broadcasting and the axis argument instead of a loop.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    
    B = A.copy()

    # The ith entry in the vstack is the sum of the ith row of B.
    # Dividing B by this vstack divides each entry in the ith row of B
    # by the ith entry in the vstack.
    return B / np.vstack(np.sum(B, axis=1))


def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid. Use slicing, as specified in the manual.
    """

    # Load the 20x20 grid.
    grid = np.load('grid.npy')

    # A dictionary that will hold the max for each direction of multiplying grid's entries.
    maxes = dict()

    '''Each grid[...] has all 20 rows, but leaves off 3 columns.
    # The first grid leaves off the last 3 columns.
    The second leaves off the first column and the last 2.
    The third leaves off the first 2 columns and the last one.
    The fourth leaves off the first 3 columns.
    Multiply these grids to get, in each position of the resulting array,
    the product of each four horizontally consecutive entries in the original grid.'''
    horiz = grid[:, :-3] * grid[:, 1:-2] * grid[:, 2:-1] * grid[:, 3:]
    maxes['horiz'] = np.max(horiz)
    
    # Each grid[...] has all 20 columns, but leaves off 3 rows,
    # in the same pattern as above for horiz.
    verti = grid[:-3, :] * grid[1:-2, :] * grid[2:-1, :] * grid[3:, :]
    maxes['verti'] = np.max(verti)

    # Each consecutive grid[...] in the product is shifted one row down and one column to the right.
    rdiag = grid[:-3, :-3] * grid[1:-2, 1:-2] * grid[2:-1, 2:-1] * grid[3:, 3:]
    maxes['rdiag'] = np.max(rdiag)

    # Each consecutive grid[...] in the product is shifted one row up and and one column to the right.
    ldiag = grid[3:, :-3] * grid[2:-1, 1:-2] * grid[1:-2, 2:-1] * grid[:-3, 3:]
    maxes['ldiag'] = np.max(ldiag)

    # The max of all the saved maxes.
    return max(maxes.values())