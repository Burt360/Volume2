# python_intro.py
"""Python Essentials: Introduction to Python.
Nathan Schill
Section 3
Thurs. Sept. 8, 2022
"""


# Problem 1 (write code below)
if __name__ == '__main__':
    print('Hello, world!')


# Problem 2
def sphere_volume(r):
    """Return the volume of the sphere of radius 'r'.
    Use 3.14159 for pi in your computation.
    """
    return 4/3 * 3.14159 * r**3


# Problem 3
def isolate(a, b, c, d, e):
    """Print the arguments separated by spaces, but print 5 spaces on either
    side of b.
    """
    print(a, ' '*5, b, ' '*5, f'{c} {d} {e}', sep='')


# Problem 4
def first_half(my_string):
    """Return the first half of the string 'my_string'. Exclude the
    middle character if there are an odd number of characters.

    Examples:
        >>> first_half("python")
        'pyt'
        >>> first_half("ipython")
        'ipy'
    """
    return my_string[:len(my_string)//2]

def backward(my_string):
    """Return the reverse of the string 'my_string'.

    Examples:
        >>> backward("python")
        'nohtyp'
        >>> backward("ipython")
        'nohtypi'
    """
    return my_string[::-1]


# Problem 5
def list_ops():
    """Define a list with the entries "bear", "ant", "cat", and "dog".
    Perform the following operations on the list:
        - Append "eagle".
        - Replace the entry at index 2 with "fox".
        - Remove (or pop) the entry at index 1.
        - Sort the list in reverse alphabetical order.
        - Replace "eagle" with "hawk".
        - Add the string "hunter" to the last entry in the list.
    Return the resulting list.

    Examples:
        >>> list_ops()
        ['fox', 'hawk', 'dog', 'bearhunter']
    """
    it = ['bear', 'ant', 'cat', 'dog']
    it.append('eagle')
    it[2] = 'fox'
    it.pop(1)
    it.sort(reverse=True)
    it[it.index('eagle')] = 'hawk'
    it[-1] += 'hunter'
    return it


# Problem 6
def pig_latin(word):
    """Translate the string 'word' into Pig Latin, and return the new word.

    Examples:
        >>> pig_latin("apple")
        'applehay'
        >>> pig_latin("banana")
        'ananabay'
    """
    vowels = 'aeiou'
    
    # First letter is a vowel.
    if word[0] in vowels:
        word += 'hay'
    # First letter is a consonant.
    else:
        first_letter = word[0]
        word = word[1:]
        word += first_letter + 'ay'
    return word


# Problem 7
def palindrome():
    """Find and retun the largest panindromic number made from the product
    of two 3-digit numbers.
    """

    '''-1 definitely isn't the largest palindromic product of two 3-digit numbers,
    and we know there is a palindromic product of two 3-digit numbers since the
    question is asking for the largest one... so it's safe to initialize the max as -1.'''
    max = -1

    # Start with the i as 999 and iterate down.
    for i in range(999, 100, -1):
        # For each i, start with j at i and iterate down.
        for j in range(i, 100, -1):
            forward_int = i*j
            forward = str(i*j)
            backward = forward[::-1]

            # Check if the int is a palindrome and is greater than the previous max.
            if forward == backward and forward_int > max:
                max = forward_int
                print(max, (i, j))
    return max

# Problem 8
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series, which approximates ln(2).
    """
    
    '''Define a generator that yields consecutive terms of the alternating harmonic series
    starting with the first term and ending with the nth (the parameter of alt_harmonic).'''
    def n_terms(n):
        for i in range(1, n+1):
            yield (-1)**(i+1)/i # 1/1, -1/2, 1/3, ...

    return(sum(n_terms(n)))


if __name__ == '__main__':
    print(sphere_volume(2))