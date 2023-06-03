from matplotlib import pyplot as plt
from time import perf_counter as pc
from pprint import pprint


############################## 4.1 ##############################

def naive_fib(n):
    """Returns the nth Fibonacci number with simple recursion."""

    if n == 0 or n == 1:
        # Base case
        return 1
    else:
        # Recurse
        return naive_fib(n-1) + naive_fib(n-2)

def memoized_fib(n, fibs={0:1, 1:1}):
    """Returns the nth Fibonacci number with memoization."""

    # Dictionary for previously computed values.

    if n == 0 or n == 1:
        # Base case
        return 1
    elif n in fibs:
        # Get the previously stored value of fib(n)
        return fibs[n]
    else:
        # Compute fib(n) and store it before returning it
        fibs[n] = memoized_fib(n-1, fibs) + memoized_fib(n-2, fibs)
        return fibs[n]

def bottom_up_fib(n):
    """Returns the nth Fibonacci number with bottom-up programming."""

    if n == 0 or n == 1:
        # fib(0) or fib(1)
        return 1

    # The previous two fib values.
    fib1 = 1
    fib2 = 1

    # Keep track of the current fib(i) being computed.
    i = 2
    while i <= n:
        # Compute fib(i), then reassign the previous two fib values.
        sum = fib1 + fib2
        fib2 = fib1
        fib1 = sum
        i += 1
        
    return sum

def q1test():
    """Test each of the three fib functions for a given value of n."""
    fns = (naive_fib, memoized_fib, bottom_up_fib)

    n = 10
    print(*[fn(n) for fn in fns])

def q1_plot():
    """Plot the times for each of the three fib functions for n = 1, ..., MAX_N."""
    
    def timefn(fn, MAX_N):
        """Return the times for each n = 1, ..., MAX_N."""

        times = [None] * MAX_N

        # Time each n
        for n in range(1, MAX_N + 1):
            start = pc()
            fn(n)
            end = pc()
            times[n-1] = end - start
        
        return times


    MAX_N = 40

    # Store the times for each function.
    fns_times = {naive_fib : None, memoized_fib : None, bottom_up_fib : None}
    for fn in fns_times:
        fns_times[fn] = timefn(fn, MAX_N)
    
    # Plot the times for each fib function for n = 1, ..., MAX_N.
    plt.plot([n for n in range(1, MAX_N + 1)], fns_times[naive_fib], label='naive_fib')
    plt.plot([n for n in range(1, MAX_N + 1)], fns_times[memoized_fib], label='memoized_fib')
    plt.plot([n for n in range(1, MAX_N + 1)], fns_times[bottom_up_fib], label='bottom_up_fib')

    # Use log scale on y-axis.
    plt.yscale('log')

    # Add labels, title, and legend, then show plot.
    plt.xlabel('n')
    plt.ylabel('time (seconds)')
    plt.title('4.1: Time to find nth Fibonacci number')
    plt.legend()
    plt.show()

def q1_find_largest_n():
    """Find the largest n for which all three functions find fib(n) in less than a minute.
    Since the naive method takes the longest, we only test that one.
    """

    """
    Result:
    [(38, 25.667229900000002), (39, 39.1025848), (40, 64.9750546)]
    """

    times = list()

    fn = naive_fib
    
    # Test n = 38, ..., 43
    for n in range(38, 44):
        start = pc()
        fn(n)
        end = pc()
        times.append((n, end - start))

        # If the previous time was 60 seconds or more, don't check more values of n.
        if times[-1][1] >= 60:
            break
    
    return times

#q1_plot()
#pprint(q1_find_largest_n())

############################## 4.2 ##############################

def naive_coins(v, C):
    """Finds the optimal number and list of coins to make change.
    
    Parameters:
        v (integer): number of cents for which to make change
        C (list): set of coin values (in cents) for coinage system
    
    Return:
        (n, o): minimum number of coins, optimal list of coins
    """
    
    # Base case
    if v == 0:
        return 0, list()

    # Get the min number of coins and optimal configurations for v-c for each c
    results = {c : naive_coins(v - c, C) for c in C if c <= v}

    # Get just the numbers of coins in order to find the best coin
    num_coins = {c : v[0] for c, v in results.items()}
    best_coin = min(num_coins, key=num_coins.get)

    # Increment the number of coins
    n = 1 + num_coins[best_coin]

    # Update the optimal configuration of coins
    optimal_config = results[best_coin][1]
    optimal_config.append(best_coin)
    
    return n, sorted(optimal_config)

def bottom_up_coins(v, C):
    """Finds the optimal number and list of coins to make change.
    
    Parameters:
        v (integer): number of cents for which to make change
        C (list): set of coin values (in cents) for coinage system
    
    Return:
        (n, o): minimum number of coins, optimal list of coins
    """
    
    # value : (min number of coins, optimal list of coins to achieve value)
    mins = {0 : (0, [])}

    a = 1 # Next value to compute min number of coins for
    while a <= v: 
        # The min number of coins for a-c for each c in C, and the coin that gets it
        vals = {mins[a-c][0] : c for c in C if a-c >= 0}
        
        # Get the min number of coins and the coin that gets it
        min_num_coins = min(vals)
        best_coin = vals[min_num_coins]

        # Store the min number of coins and the optimal list of coins to get it
        mins[a] = (1 + min_num_coins, mins[a-best_coin][1] + [best_coin])

        a += 1
    
    return mins[v][0], sorted(mins[v][1])

US_C = [1, 5, 10, 25, 50, 100]
#print(naive_coins(19, US_C))
#print(bottom_up_coins(190, US_C))

############################## 4.2-3 ##############################

def greedy_coins(v, C):
    """Finds the optimal number and list of coins to make change.
    
    Parameters:
        v (integer): number of cents for which to make change
        C (list): set of coin values (in cents) for coinage system
    
    Return:
        (n, o): minimum number of coins, optimal list of coins
    """
    
    optimal_list = list()

    running_total = 0
    while running_total < v:
        # Append the largest coin that doesn't cause the running total to exceed the value
        optimal_list.append(max([c for c in C if running_total + c <= v]))

        # Add the coin found above to the running total
        running_total += optimal_list[-1]
        
    return len(optimal_list), sorted(optimal_list)

#print(greedy_coins(19, US_C))

def q2_3_plot():
    """Plot the times for each of the three coins functions for n = 1, ..., MAX_N."""
    
    def timefn(fn, C, MAX_N):
        """Return the times for each n = 1, ..., MAX_N."""

        times = [None] * MAX_N
        
        # Time each n
        MAX_SECONDS = 60
        for n in range(1, MAX_N + 1):
            start = pc()
            fn(n, C)
            end = pc()
            times[n-1] = end - start
            if times[n-1] >= MAX_SECONDS:
                break
        
        return times


    US_C = [1, 5, 10, 25, 50, 100]
    MAX_N = 1999
    
    # Store the times for each function.
    fns_times = {naive_coins : None, bottom_up_coins : None, greedy_coins : None}
    #fns_times = {bottom_up_coins : None, greedy_coins : None}
    for fn in fns_times:
        fns_times[fn] = timefn(fn, US_C, MAX_N)
    
    # 4.2: Plot the times for each coins function (except greedy) for n = 1, ..., MAX_N.
    subplot = plt.subplot(121)
    subplot.plot([n for n in range(1, len(fns_times[naive_coins]) + 1)], fns_times[naive_coins], label='naive_coins')
    subplot.plot([n for n in range(1, MAX_N + 1)], fns_times[bottom_up_coins], label='bottom_up_coins')
    
    # Set labels, log scale, title, and activate legend.
    subplot.set_xlabel('n')
    subplot.set_yscale('log')
    subplot.legend()
    subplot.set_title('4.2: Time to find change for n cents')

    subplot.set_ylabel('time (seconds)')

    # 4.3: Plot the times for each coins function for n = 1, ..., MAX_N.
    subplot = plt.subplot(122)
    subplot.plot([n for n in range(1, MAX_N + 1)], fns_times[naive_coins], label='naive_coins')
    subplot.plot([n for n in range(1, MAX_N + 1)], fns_times[bottom_up_coins], label='bottom_up_coins')
    subplot.plot([n for n in range(1, MAX_N + 1)], fns_times[greedy_coins], label='greedy_coins')

    # Set labels, log scale, title, and activate legend.
    subplot.set_xlabel('n')
    subplot.set_yscale('log')
    subplot.legend()
    subplot.set_title('4.3: Time to find change for n cents')

    # Show plot.
    plt.tight_layout()
    plt.show()

#q2_3_plot()

def compare_bottom_up_and_greedy_coins():
    """Check if greedy gets the optimal solution for the US coinage system."""

    """Result: greedy does in fact return the optimal solution for the US coinage system"""

    US_C = [1, 5, 10, 25, 50, 100]
    MAX_N = 1999

    for n in range(1, MAX_N + 1):
        # If the optimal solution is not the same as the greedy solution, return the n and the two solutions.
        if (bottom := bottom_up_coins(n, US_C)[1]) != (greedy := greedy_coins(n, US_C)[1]):
            return n, bottom, greedy
        else:
            return -1

#print(compare_bottom_up_and_greedy_coins())