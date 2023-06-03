from markov_chains import *
import pytest

import numpy as np

def test_MarkovChain():
    # Non-square matrix
    C = np.array([[0.25, 0.3, 0.35],[0.75, 0.7, 0.65]])
    with pytest.raises(ValueError):
        MarkovChain(C)
    
    # Non-column stochastic matrix
    D = np.array([[0.25, 0.75], [0.5, 0.5]])
    with pytest.raises(ValueError):
        MarkovChain(D)
    
    E = np.array([[0.25, 0.4], [0.75, 0.6]])
    mc = MarkovChain(E)

    A = np.array([[0.7, 0.6], [0.3, 0.4]])
    labels = ['hot', 'cold']
    hotcold = MarkovChain(A, labels)

    assert hotcold.labels == labels
    assert hotcold.labelmap == {'hot' : 0, 'cold' : 1}


    assert hotcold.transition('hot') in labels
    print(hotcold.transition('hot'))
    assert hotcold.transition('cold') in labels
    print(hotcold.transition('cold'))

    ### Test walk
    N = 5
    walk = hotcold.walk('hot', N)
    assert walk[0] == 'hot'
    assert len(walk) == N
    print(walk)

    N = 5
    walk = hotcold.walk('cold', N)
    assert walk[0] == 'cold'
    assert len(walk) == N
    print(walk)

    ### Test path
    path = hotcold.path('hot', 'hot')
    assert path[0] == 'hot'
    assert len(path) == 1
    print(path)

    path = hotcold.path('hot', 'cold')
    assert path[0] == 'hot'
    assert path[-1] == 'cold'
    assert len(path) >= 2
    print(path)

    print()
    # Four-state weather model
    B = np.array([[0.5, 0.3, 0.1, 0], [0.3, 0.3, 0.3, 0.3], [0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.2]])
    labels = ['hot', 'mild', 'cold', 'freezing']
    fourstate = MarkovChain(B, labels)
    
    print('Four-state weather model\n')

    N = 10
    walk = fourstate.walk('cold', N)
    assert walk[0] == 'cold'
    assert len(walk) == N
    print(walk)

    ### Test path
    path = fourstate.path('hot', 'freezing')
    assert path[0] == 'hot'
    assert path[-1] == 'freezing'
    assert len(path) >= 2
    print(path)

def test_steady_state():
    WALK_LENGTH = 1000

    # Two-state weather model
    A = np.array([[0.7, 0.6], [0.3, 0.4]])
    labels = ['hot', 'cold']
    hotcold = MarkovChain(A, labels)

    hotcold_steady_state = hotcold.steady_state()
    assert np.allclose(A@hotcold_steady_state, hotcold_steady_state)
    assert np.allclose(np.linalg.matrix_power(A, 10), np.array([[2/3, 2/3], [1/3, 1/3]]))
    
    hotcold_walk = hotcold.walk(np.random.choice(labels), WALK_LENGTH)
    print('hotcold: steady state | walk proportion')
    print('hot:', hotcold_steady_state[0], '|', [state == 'hot' for state in hotcold_walk].count(True)/WALK_LENGTH)
    print('cold:', hotcold_steady_state[1], '|', [state == 'cold' for state in hotcold_walk].count(True)/WALK_LENGTH)
    print()

    # Four-state weather model
    B = np.array([[0.5, 0.3, 0.1, 0], [0.3, 0.3, 0.3, 0.3], [0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.2]])
    labels = ['hot', 'mild', 'cold', 'freezing']
    fourstate = MarkovChain(B, labels)

    fourstate_steady_state = fourstate.steady_state()
    assert np.allclose(B@fourstate_steady_state, fourstate_steady_state)

    fourstate_walk = fourstate.walk(np.random.choice(labels), WALK_LENGTH)
    print('fourstate: steady state | walk proportion')
    print('hot:', fourstate_steady_state[0], '|', [state == 'hot' for state in fourstate_walk].count(True)/WALK_LENGTH)
    print('mild:', fourstate_steady_state[1], '|', [state == 'mild' for state in fourstate_walk].count(True)/WALK_LENGTH)
    print('cold:', fourstate_steady_state[2], '|', [state == 'cold' for state in fourstate_walk].count(True)/WALK_LENGTH)
    print('freezing:', fourstate_steady_state[3], '|', [state == 'freezing' for state in fourstate_walk].count(True)/WALK_LENGTH)
    print()


    for n in range(2, 5):
        C = np.random.random((n,n))
        C /= C.sum(axis=0)
        MC = MarkovChain(C)
        try:
            steady_state = MC.steady_state()
        except ValueError as err:
            print(f'size {n}:', err)
        else:
            assert np.allclose(C@steady_state, steady_state)
            assert np.allclose(np.linalg.matrix_power(C, 40)[:,0], steady_state)

def test_babble():
    yoda = SentenceGenerator('yoda.txt')
    
    for _ in range(10):
        print(yoda.babble())