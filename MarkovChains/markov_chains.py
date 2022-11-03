# markov_chains.py
"""Volume 2: Markov Chains.
Nathan Schill
Sec. 3
Thurs. Nov. 10, 2022
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        
        # Check if A is not square
        if A.shape[0] != A.shape[1]:
            raise ValueError('matrix not square')
        
        # Check if A is not column stochastic
        if not np.allclose(A.sum(axis=0), np.ones(A.shape[1])):
            raise ValueError('matrix not column stochastic')
        
        # Store A
        self.m = A

        self.labelmap = dict()
        if states is not None:
            # If given state labels, store them and create the label-to-index dict
            self.labels = states
            for i, label in enumerate(states):
                self.labelmap[label] = i
        else:
            # Otherwise, store labels 0, ..., n-1 and create the label-to-index dict
            self.labels = [i for i in range(A.shape[0])]
            for i in self.labels:
                self.labelmap[i] = i


    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        
        # Get index of given state
        in_index = self.labelmap[state]

        # Get index of state transitioned to
        out_index = np.argmax(np.random.multinomial(1, self.m[:, in_index]))

        # Return state transitioned to
        return self.labels[out_index]


    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """

        # Raise a ValueError if start is not a valid state
        if start not in self.labelmap:
            raise ValueError('start not a valid label')

        # Init walk with start state
        walk = [start]

        # Transition N-1 times
        for i in range(N-1):
            walk.append(self.transition(walk[-1]))

        return walk

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        
        # Raise a ValueError if either start or stop is not a valid state
        if start not in self.labelmap or stop not in self.labelmap:
            raise ValueError('start or stop not a valid label')

        # Init path with start state
        path = [start]

        # Transition until reaching stop state
        while path[-1] != stop:
            path.append(self.transition(path[-1]))

        return path


    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        raise NotImplementedError("Problem 4 Incomplete")

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        raise NotImplementedError("Problem 6 Incomplete")
