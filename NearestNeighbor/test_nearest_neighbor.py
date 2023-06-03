import pytest
from nearest_neighbor import *

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree

def test_exhaustive_search():
    X = np.array([[1,2,3], [4,5,6], [7,8,9]])
    z = np.array([1,2,3])

    x, d = exhaustive_search(X, z)
    assert np.allclose(x, [1,2,3])
    assert np.isclose(d, la.norm(x - z))

    X = np.array([[5,5], [3,2], [8,4], [2,6], [7,7]])
    z = np.array([3,2.75])

    x, d = exhaustive_search(X, z)
    assert np.allclose(x, np.array([3,2]))
    assert np.isclose(d, la.norm(x - z))

def test_KDTNode():
    z = 1

    with pytest.raises(TypeError):
        KDTNode(z)
    
    x = np.array([1,2,3])
    node = KDTNode(x)
    assert np.allclose(node.value, x)

def test_insert():
    tree = KDT()
    X = np.array([[3,1,4], [1,2,7], [4,3,5], [2,0,3], [2,4,5], [6,1,4], [1,4,3], [0,5,7], [5,2,5]])
    
    tree.insert(X[0])
    assert np.allclose(tree.find(X[0]).value, X[0])

    print(tree, '\n')

    with pytest.raises(ValueError):
        # Inserting a 4-d vector into 3-d tree
        tree.insert(np.array([1,2,3,4]))
    
    for z in X[1:,:]:
        # Insert all but the first row vector in X
        tree.insert(z)
    
    assert np.allclose(tree.find(X[8]).value, X[8])

    with pytest.raises(ValueError):
        # Inserting a duplicate into the tree
        tree.insert(X[8])

    print(tree, '\n')

def test_query():
    tree = KDT()
    X = np.array([[5,5], [3,2], [8,4], [2,6], [7,7]])

    for z in X:
        # Insert all row vectors in X
        tree.insert(z)
    
    target = np.array([3,2.75])
    x, d = tree.query(target)

    assert np.allclose(x, np.array([3,2]))
    assert d, 0.75


    search = np.array([5,3,7])

    X = np.array([[3,1,4], [1,2,7], [4,3,5], [2,0,3], [2,4,5], [6,1,4], [1,4,3], [0,5,7], [5,2,5]])
    tree = KDT()
    for z in X:
        tree.insert(z)
    
    test_tree = KDTree(X)

    x, d = tree.query(search)

    dp, i = test_tree.query(search)
    xp = test_tree.data[i]

    assert np.allclose(x, xp)
    assert d == dp

    xe, de = exhaustive_search(X, search)
    assert np.allclose(x, xe)
    assert d == de


'''
def test_KNC():
    """Test KNeighborsClassifier against scipy's version.
    [Put this function in the main file, not the test file run by pytest.]
    """

    from sklearn.neighbors import KNeighborsClassifier as KNC
    
    NUM_NODES = 100
    DIMENSION = 5
    NUM_LABELS = 5

    DATA = np.random.random((NUM_NODES, DIMENSION))
    LABELS = np.random.randint(0, NUM_LABELS-1, size=NUM_NODES)
    TARGET = np.random.random(DIMENSION)

    tested = KNeighborsClassifier(DIMENSION)
    tester = KNC(DIMENSION)

    tested.fit(DATA, LABELS)
    tester.fit(DATA, LABELS)

    print(tested.predict(TARGET), tester.predict(TARGET.copy().reshape(1,-1))[0])
'''
