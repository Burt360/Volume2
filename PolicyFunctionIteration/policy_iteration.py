# policy_iteration.py
"""Volume 2: Policy Function Iteration.
Nathan Schill
Section 2
Thurs. Apr. 20, 2023
"""

import numpy as np

import gym
from gym import wrappers

# Intialize P for test example
# Left = 0
# Down = 1
# Right = 2
# Up= 3

P = {s : {a: [] for a in range(4)} for s in range(4)}
P[0][0] = [(0, 0, 0, False)]
P[0][1] = [(1, 2, -1, False)]
P[0][2] = [(1, 1, 0, False)]
P[0][3] = [(0, 0, 0, False)]
P[1][0] = [(1, 0, -1, False)]
P[1][1] = [(1, 3, 1, True)]
P[1][2] = [(0, 0, 0, False)]
P[1][3] = [(0, 0, 0, False)]
P[2][0] = [(0, 0, 0, False)]
P[2][1] = [(0, 0, 0, False)]
P[2][2] = [(1, 3, 1, True)]
P[2][3] = [(1, 0, 0, False)]
P[3][0] = [(0, 0, 0, True)]
P[3][1] = [(0, 0, 0, True)]
P[3][2] = [(0, 0, 0, True)]
P[3][3] = [(0, 0, 0, True)]


## Problem 1
def value_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=3000):
    """Perform Value Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
       v (ndarray): The discrete values for the true value function.
       n (int): number of iterations
    """

    v_old = np.zeros(nS)
    v_new = np.copy(v_old)
    
    # Iterate through v_k
    for i in range(maxiter):

        # Iterate through states to populate v_new from v_old
        for s in P:

            # Iterate through actions
            sa_vector = np.zeros(nA)
            for a in range(nA):

                # Iterate through possible next states for given action
                for tuple_info in P[s][a]:

                    # tuple_info is a tuple of (probability, next state, reward, done)
                    p, s_, u, _ = tuple_info

                    # Sum up the possible end states and rewards with given action
                    sa_vector[a] += p * (u + beta * v_old[s_])
            
            # Add the max value to the value function
            v_new[s] = np.max(sa_vector)
        
        # Check tolerance
        if np.linalg.norm(v_new - v_old) < tol:
            break
        
        # Set new v_old
        v_old = np.copy(v_new)

    return v_new, i+1


# Problem 2
def extract_policy(P, nS, nA, v, beta=1.0):
    """Returns the optimal policy vector for value function v

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        v (ndarray): The value function values.
        beta (float): The discount rate (between 0 and 1).

    Returns:
        policy (ndarray): which direction to move in from each square.
    """
    
    # Policy vector
    c = np.empty(nS, dtype=int)

    # Iterate through states
    for s in range(nS):

        # Iterate through actions
        sa_vector = np.zeros(nA)
        for a in range(nA):

            # Iterate through possible next states for given action
            for tuple_info in P[s][a]:

                # tuple_info is a tuple of (probability, next state, reward, done)
                p, s_, u, _ = tuple_info

                # Sum up the possible end states and rewards with given action
                sa_vector[a] += p * (u + beta * v[s_])
        
        # Record the argmax in the policy
        c[s] = np.argmax(sa_vector)

    return c


# Problem 3
def compute_policy_v(P, nS, nA, policy, beta=1.0, tol=1e-8, maxiter=3000):
    """Computes the value function for a policy using policy evaluation.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        policy (ndarray): The policy to estimate the value function.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.

    Returns:
        v (ndarray): The discrete values for the true value function.
    """

    v_old = np.zeros(nS)
    v_new = np.copy(v_old)
    
    # Iterate through v_k
    for i in range(maxiter):

        # Iterate through states to populate v_new from v_old
        for s in P:
            
            # Choose action with given policy
            a = policy[s]
            
            sa_value = 0
            
            # Iterate through possible next states for given action
            try:
                P[s][a]
            except:
                print(a)
                print(policy)
            for tuple_info in P[s][a]:

                # tuple_info is a tuple of (probability, next state, reward, done)
                p, s_, u, _ = tuple_info

                # Sum up the possible end states and rewards with given action
                sa_value += p * (u + beta * v_old[s_])
            
            # Record the value in the value function
            v_new[s] = sa_value

        # Check tolerance
        if np.linalg.norm(v_new - v_old) < tol:
            break
        
        # Set new v_old
        v_old = np.copy(v_new)

    return v_new


# Problem 4
def policy_iteration(P, nS, nA, beta=1, tol=1e-8, maxiter=200):
    """Perform Policy Iteration according to the Bellman optimality principle.

    Parameters:
        P (dict): The Markov relationship
                (P[state][action] = [(prob, nextstate, reward, is_terminal)...]).
        nS (int): The number of states.
        nA (int): The number of actions.
        beta (float): The discount rate (between 0 and 1).
        tol (float): The stopping criteria for the value iteration.
        maxiter (int): The maximum number of iterations.

    Returns:
    	v (ndarray): The discrete values for the true value function
        policy (ndarray): which direction to move in each square.
        n (int): number of iterations
    """
    
    # Init random policy
    p0 = np.random.choice(nA, size=nS)
    p1 = p0.copy()
    
    for i in range(maxiter):
        v = compute_policy_v(P, nS, nA, p0)
        p1 = extract_policy(P, nS, nA, v)

        # Check tolerance
        if np.linalg.norm(p1 - p0) < tol:
            break
        
        p0 = p1.copy()

    return v, p0, i


# Problem 5 and 6
def frozen_lake(basic_case=True, M=1000, render=False):
    """Finds the optimal policy to solve the FrozenLake problem.

    Parameters:
    basic_case (boolean): True for 4x4 and False for 8x8 environemtns.
    M (int): The number of times to run the simulation using problem 6.
    render (boolean): Whether to draw the environment.

    Returns:
    vi_policy (ndarray): The optimal policy for value iteration.
    vi_total_rewards (float): The mean expected value for following the value iteration optimal policy.
    pi_value_func (ndarray): The maximum value function for the optimal policy from policy iteration.
    pi_policy (ndarray): The optimal policy for policy iteration.
    pi_total_rewards (float): The mean expected value for following the policy iteration optimal policy.
    """

    if basic_case:
        # Make environment for 4x4 scenario
        env_name = 'FrozenLake-v1'
    else:
        # Make environment for 8x8 scenario
        env_name = 'FrozenLake8x8-v1'
    
    env = gym.make(env_name, new_step_api=True).env

    # Find number of states and actions
    nS = env.observation_space.n
    nA = env.action_space.n
    
    # Get the dictionary with all the states and actions
    P = env.P

    # Value iteration
    vi_value_func = value_iteration(P, nS, nA)[0]
    vi_policy = extract_policy(P, nS, nA, vi_value_func)

    # Policy iteration
    (pi_value_func, pi_policy) = policy_iteration(P, nS, nA)[:2]
    
    # Render once with value iteration's policy and once with policy iteration's policy
    if render:
        run_simulation(env, vi_policy, True)
        run_simulation(env, pi_policy, True)

    vi_total_rewards = 0
    pi_total_rewards = 0
    # Simulate both policies
    for _ in range(M):
        vi_total_rewards += run_simulation(env, vi_policy, False)
        pi_total_rewards += run_simulation(env, pi_policy, False)
    
    pi_total_rewards /= M
    vi_total_rewards /= M
    return vi_policy, vi_total_rewards, pi_value_func, pi_policy, pi_total_rewards


# Problem 6
def run_simulation(env, policy, render=False, beta=1.0):
    """ Evaluates policy by using it to run a simulation and calculate the reward.

    Parameters:
    env (gym environment): The gym environment.
    policy (ndarray): The policy used to simulate.
    beta float: The discount factor.
    render (boolean): Whether to draw the environment.

    Returns:
    total reward (float): Value of the total reward received under policy.
    """
    
    # Put environment in starting state
    obs = env.reset()

    done = False
    while not done:
        # Take a step in the optimal direction and update variables
        obs, reward, done, _, _ = env.step(int(policy[obs]))

    # When done, the reward is either 1 for success or 0 for failure
    return reward