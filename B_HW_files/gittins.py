import numpy as np

def pull(probabilities, payouts, action):
    """Simulate one pull.
    
    Return:
        payout (float)
        state_update (ndarray): [1,0] or [0,1]
    """

    # Sample from Bernoulli
    payed = np.random.binomial(1, probabilities[action])

    # Return payout, (sucess, failure) pair
    return payed * payouts[action], \
        np.array([1,0]) if payed else np.array([0,1])


def compute_R(r, M, beta):
    """Compute values of R(a,b,r) for 1 <= a+b <= M with discount beta."""

    # Init
    R_values = np.zeros((M+1, M+1))

    # Each [a, b] entry (one-indexed) contains a+b
    index_sums = np.zeros_like(R_values)
    A = np.arange(M+1)
    B = np.arange(M+1)[:, np.newaxis]
    index_sums += A + B
    
    # Only modify entries where a+b = M
    M_mask = (index_sums == M)

    # Use equation (17.10) in Volume 2
    R_values[M_mask] = np.maximum(A/index_sums[M_mask], r)/(1-beta)
    
    # Recurse with equation (17.7)
    for i in range(1, M):
        index_slice = index_sums[:-i, :-i]
        index_mask = (index_slice == M-i)
        
        R_values[:-i, :-i][index_mask] = \
            np.maximum( \
                (
                    (
                        A[-i-1::-1] * (1 + beta*R_values[A[:-i]+1, :-i]) + \
                        A[:-i]      * (    beta*R_values[:-i, A[:-i]+1])
                    ) / (M-i) \
                )[index_mask],
                r / (1-beta)
            )

    return R_values


def gittins(payouts, states, M, K, beta, all_R_values=None):
    """Compute approximate Gittins index for each arm.

    Assume n arms
    
    Args:
        payouts ((n,)-ndarray, dtype float)
        states ((n,2)-ndarray, dtype int): [[a1,b1], ..., [an, bn]]
        M (int): use for compute_R
        K (int): grid size to find r which most nearly approximates
                 Gittins index for each arm
        beta (float): discount factor
        all_R_values: after first run of this function, get all_R_values
                      from return and continue using
    Return:
        index ((n,)-ndarray, dtype float): array of Gittins indices
        all_R_values: allow to use again on next call to this function
    """

    # Grid of K possible values of r
    r = np.linspace(0, 1, K+1, endpoint=False)[1:]

    if all_R_values is None:
        # Compute R(a,b,r) for r in R and 1 <= a+b <= M
        all_R_values = np.stack(np.vectorize(compute_R, otypes=[np.ndarray])(r, M, beta))
    
    # Gittins index for each state
    indices = -1 * np.ones_like(payouts)

    # For each state, find approximate index using equation (17.9)
    for i, (a, b) in enumerate(states):
        ab = a+b
        # If a+b == M, then use a and b rather than a+1 and b+1
        if ab >= M:
            # If a+b > M, nudge a and b down so that a+b = M
            if ab > M:
                # If odd difference, nudge a down so even difference
                if (ab-M) % 2 == 1:
                    a -= 1
                
                a -= int((ab - M)/2)
                b -= int((ab - M)/2)

                # If a or b ends up being negative, nudge it up and other down
                if a < 0:
                    b += a
                    a = 0
                elif b < 0:
                    a += b
                    b = 0
                    
            R_values = all_R_values[:,a,b]
            right_side = (1-beta) * (a*(1 + beta*R_values) + b*beta*R_values) / (a+b)
            
            # Get value of r that minimizes the absolute difference
            indices[i] = np.argmin(np.abs(r - right_side))

        elif 1 <= a+b < M:
            right_side = (1-beta) * \
                    (a*(1 + beta*all_R_values[:,a+1,b]) + b*beta*all_R_values[:,a,b+1]) \
                / (a+b)
            
            indices[i] = np.argmin(np.abs(r - right_side))

    return indices * payouts, all_R_values


def simulation(probabilities, payouts, K, T, M, beta):
    """Simulate bandit problem using Gittins index.
    
    Args:
        K (float): 0 < K < 1
        T (int): number of iterations
        M (int): M > T
        beta (float): discount factor
    
    Return:
        total_reward (float)
        estimated_probabilities (ndarray)
    """

    n = len(probabilities)

    # Init states
    states = np.array([[1,1] for _ in range(n)])

    # First run
    indices, all_R_values = gittins(payouts, states, M, K, beta)
    choice = np.argmax(indices)
    total_reward, state_update = pull(probabilities, payouts, choice)
    states[choice] += state_update

    # T-1 runs
    for i in range(T-1):

        indices, _ = gittins(payouts, states, M, K, beta, all_R_values)
        choice = np.argmax(indices)

        reward, state_update = pull(probabilities, payouts, choice)

        total_reward += beta**i * reward
        states[choice] += state_update

    # Estimate probabilities
    estimated_probabilities = states[:,0] / np.sum(states, axis=1)

    return total_reward, estimated_probabilities