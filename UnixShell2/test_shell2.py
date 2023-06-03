from shell2 import *
import pytest

# problem 6
# awk ' BEGIN{ FS = "\t" }; {print $7" "$9} ' < files.txt | sort -r > date_modified.txt

def test_grep():
    print(grep('range', '*.py'))

def test_largest():
    n = 7
    print(largest_files(n))