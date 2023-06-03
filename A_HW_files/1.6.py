def subtract(a, b):
    """
    Subtract b from a. Return the resulting difference as a list.
    b should be less than or equal to a.

    Params:
        a (list): a list of n positive single-digit integers, where each integer
                  represents a digit in the number a.
        b (list): a list of <= n positive single-digit integers, as in a.
    """

    # a is assumed to be greater than or equal to b, so append zeros (if necessary)
    # to the leading end of b so that a and b have the same length.
    delta = len(a) - len(b)
    b = delta * [0] + b

    # Iterate from right to left along a.
    for i in range(len(a) - 1, -1, -1):
        # If the digit in a is less than the digit in b,
        # subtract 1 from the next digit to the left of a and add 10 to the current digit of a.
        if a[i] < b[i]:
            a[i-1] -= 1
            a[i] += 10
        # Subtract the digit in b from the digit in a.
        a[i] -= b[i]
    
    return a