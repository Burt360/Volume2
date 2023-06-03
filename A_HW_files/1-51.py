def my_euclid(c, d):
    """Extended Euclidean algorithm:
        return gcd(a, b), x, y such that gcd(a, b) = ax + by.
    """

    if c == 0:
        return d, 0, 1
    elif d == 0:
        return c, 1, 0
    elif c == 0 and d == 0:
        return 0, 0, 0

    reversed = False
    if c >= d:
        a = c
        b = d
    else:
        a = d
        b = c
        reversed = True

    def gcd(a, b):
        r = a%b
        c = a//b
        #print(f'{a} = {c}*{b} + {r}')

        if r == 0:
            #print()
            return b, 0, 1
        else:
            g, x, y = gcd(b, r)

            tmp = x
            x = y
            y = tmp + y*-c

            print(f'{g} = {x}*{a} + {y}*{b}')
            #print(c, x, y)

            return g, x, y
        
    g, x, y = gcd(a, b)
    if not reversed:
        return g, x, y
    else:
        return g, y, x