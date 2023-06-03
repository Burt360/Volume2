from interior_point_linear import *

def test1():
    np.random.seed(42)
    
    print()
    j, k = 7, 5
    A, b, c, x = randomLP(j, k)
    point, value = interiorPoint(A, b, c)
    assert np.allclose(x, point[:k])
    # print(x, point)

def test_LAD():
    # return
    leastAbsoluteDeviations()