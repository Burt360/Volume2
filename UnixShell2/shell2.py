# shell2.py
"""Volume 3: Unix Shell 2.
Nathan Schill
Section 3
Thurs. Nov. 17, 2022
"""

from glob import glob
import subprocess


# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.

    Returns:
        matched_files (list): list of the filenames that matched the file
               pattern AND the target string.
    """
    
    # Get the list of filenames that match file_pattern
    filename_matches = glob(f'**/{file_pattern}', recursive=True)

    # Get a list of 0s and 1s (0 means grep matched target_string in filename, 1 if not)
    matches = list()
    for filename in filename_matches:
        try:
            subprocess.check_output(['grep', target_string, filename]).decode()

            # Exit code 0: match
            matches.append(filename)
        except subprocess.CalledProcessError:
            # Exit code 1: no match
            pass

    return matches


# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    
    filenames = glob('**/*.*', recursive=True)

    files_sizes = dict()
    for filename in filenames:
        output = subprocess.check_output(['ls', '-s', filename]).decode()

        # Store the size (take the characters up to the first space)
        files_sizes[filename] = output[:output.find(' ')]
    
    # Get a list of the filenames sorted from largest to smallest
    sorted_files = sorted(files_sizes, key=files_sizes.get, reverse=True)
    
    # Get n largest files
    n_largest = sorted_files[:n]

    # Get the line count of the smallest file (and remove the name of the file that is also output)
    line_count_smallest = subprocess.check_output(['wc', '-l', n_largest[-1]]).decode()
    line_count_smallest = line_count_smallest[:line_count_smallest.find(' ')]

    # Write to a file the line count found above
    with open('smallest.txt', 'w') as file:
        file.write(line_count_smallest)

    return n_largest
    
    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter
