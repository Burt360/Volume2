# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
Nathan Schill
Section 3
Thurs. Oct. 27
"""

import numpy as np
from scipy import linalg as la

#from sklearn.neighbors import KNeighborsClassifier as KNC


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    
    # For each row vector in X, get the distance from z.
    distances = la.norm(X - z, axis=1)

    # Get the index in distances (and therefore X) of the
    # row vector in X with the smallest distance from z.
    index_nearest = np.argmin(distances)

    # Return the corresponding row vector and distance from z.
    return X[index_nearest], distances[index_nearest]
    


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """

    def __init__(self, x):
        """
        Raise a TypeError if x is not a np.ndarray

        Param:
            x (np.ndarray): a vector in R^k
        """
        # Check the type of x
        if not isinstance(x, np.ndarray):
            raise TypeError('x should be a np.ndarray')
        
        # Init value
        self.value = x

        # Init node attributes
        self.left = None
        self.right = None
        self.pivot = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        
        if self.root is None:
            # If the tree is empty:
            # Set the new node as the root, set the new node's pivot as 0
            self.root = KDTNode(data)
            self.root.pivot = 0

            # Set the tree's k attribute as the length of the new node's data
            self.k = len(data)
            return
        
        if len(data) != self.k:
            # Raise a ValueError if the new data does not match the k dimensionality of the tree
            raise ValueError(f'data should be {self.k}-d')

        def _step(current, parent, left):
            """Recursively step through the tree until finding where the new node
            containing the data should go.
            If a node containing the data is already present, raise a ValueError.

            Params:
                current: the node to check
                parent: the parent of the node to check
                left (bool): True if current is parent's left child, False otherwise
            """
            if current is None:
                # Base case 1: this is where the node should go
                current = KDTNode(data)

                # Put the new node on the correct side of parent
                if left:
                    parent.left = current
                else:
                    parent.right = current
            
                # Set the new node's pivot
                if parent.pivot == self.k - 1:
                    current.pivot = 0
                else:
                    current.pivot = parent.pivot + 1
                
            elif np.allclose(data, current.value):
                # Base case 2: data is already in the tree
                raise ValueError(str(data) + ' is already in the tree')

            elif data[current.pivot] < current.value[current.pivot]:
                # Recursively search left
                return _step(current.left, current, True)
            else:
                # Recursively search right
                return _step(current.right, current, False)         

        # Start the recursive search at the root of the tree
        _step(self.root, None, None)


    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        
        if self.root is None:
            raise ValueError('tree is empty')

        ### Using Algorithm 3.1 from NearestNeighbor PDF

        def _search(current, nearest, d):
            if current is None:
                return nearest, d
            
            x = current.value
            i = current.pivot

            if (d_check := la.norm(x - z)) < d:
                nearest = current
                d = d_check
            
            if z[i] < x[i]:
                nearest, d = _search(current.left, nearest, d)
                if z[i] + d >= x[i]:
                    nearest, d = _search(current.right, nearest, d)
            else:
                nearest, d = _search(current.right, nearest, d)
                if z[i] - d <= x[i]:
                    nearest, d = _search(current.left, nearest, d)
            
            return nearest, d
        
        node, d = _search(self.root, self.root, la.norm(self.root.value - z))

        return node.value, d


    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """

    

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    raise NotImplementedError("Problem 6 Incomplete")
