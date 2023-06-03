from dynamic_programming import *

def test1():
    assert calc_stopping(4) == (0.4583333333333333, 1)
    # print(calc_stopping(4))

def test2():
    # return
    assert graph_stopping_times(1000) == 0.368

def test3():
    return
    print()
    print(get_consumption(4, lambda x: x))

def test4():
    return
    print()
    print(eat_cake(3, 4, 0.9))

    # Alternative way of populating A
    # A[:,t] = np.max(CVt, axis=1)

def test7():
    return
    print()
    print(find_policy(3, 4, 0.9))