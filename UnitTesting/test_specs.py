# test_specs.py
"""Python Essentials: Unit Testing.
Nathan Schill
Section 3
Thurs. Sept. 29, 2022
"""

import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
        assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    """Test various cases for specs.smallest_factor."""

    # Assign the function to a shorter variable name.
    t = specs.smallest_factor

    ### We'll consider the string messages following asserts as 'comments.'

    assert t(1) == 1, 'spf of 1'
    assert t(2) == 2, 'spf of 2'
    assert t(3) == 3, 'spf of 3'
    assert t(4) == 2, 'spf of 4'
    assert t(0) == 0, 'spf of 0'
    assert t(10) == 2, 'spf of 10'

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    """Test various cases for specs.month_length."""
    
    # Assign the function to a shorter variable name.
    t = specs.month_length

    ### We'll consider the string messages following asserts as 'comments.'

    # Test leap year on February and non-February month.
    assert t('January') == 31, 'Jan -> 31'
    assert t('January', True) == 31, 'Jan leap -> 31'

    assert t('February') == 28, 'Feb -> 28'
    assert t('February', True) == 29, 'Feb leap -> 29'

    # Test all the other months.
    assert t('March') == 31, 'Mar -> 31'
    assert t('April') == 30, 'Apr -> 30'
    assert t('May') == 31, 'May -> 31'
    assert t('June') == 30, 'Jun -> 30'
    assert t('July') == 31, 'Jul -> 31'
    assert t('August') == 31, 'Aug -> 31'
    assert t('September') == 30, 'Sep -> 30'
    assert t('October') == 31, 'Oct -> 31'
    assert t('November') == 30, 'Nov -> 30'
    assert t('December') == 31, 'Dec -> 31'

    # Test a non-month argument.
    assert t('not_a_month') == None, 'other -> None'

# Problem 3: write a unit test for specs.operate().
def test_operate():
    """Test various cases for specs.operate."""
    
    # Assign the function to a shorter variable name.
    t = specs.operate

    ### We'll consider the string messages following asserts as 'comments.'

    # Test operator.
    with pytest.raises(TypeError) as excinfo:
        t(1, 1, 1)
        assert excinfo.value.args[0] == 'oper must be a string'
    with pytest.raises(ValueError) as excinfo:
        t(1, 1, 'plus')
        assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"
    
    # Test operands.
    assert t(1, 1, '+') == 2, '1 + 1'
    assert t(1, 1, '-') == 0, '1 - 1'
    assert t(2, 3, '*') == 6, '2 * 3'
    assert t(6, 3, '/') == 2, '6 / 3'

    # Test division by zero.
    with pytest.raises(ZeroDivisionError) as excinfo:
        t(1, 0, '/')
        assert excinfo.value.args[0] == 'division by zero is undefined'
    

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    # Assign the function to a shorter variable name.
    frac = specs.Fraction

    frac_1_3 = frac(1, 3)
    frac_1_2 = frac(1, 2)
    frac_n2_3 = frac(-2, 3)
    frac_2_1 = frac(2, 1)

    return frac_1_3, frac_1_2, frac_n2_3, frac_2_1

def test_fraction_init(set_up_fractions):
    """Test various cases for specs.Fraction.__init__."""
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac_30_42 = specs.Fraction(30, 42)
    assert frac_30_42.numer == 5
    assert frac_30_42.denom == 7

    # Assign the function to a shorter variable name.
    frac = specs.Fraction

    # Test that __init__ raises the correct errors.
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_1_0 = frac(1, 0)
        assert excinfo.value.args[0] == 'denominator cannot be zero'
    with pytest.raises(TypeError) as excinfo:
        frac_float_1 = frac(0.5, 1)
        assert excinfo.value.args[0] == 'numerator and denominator must be integers'
    with pytest.raises(TypeError) as excinfo:
        frac_1_float = frac(1, 0.5)
        assert excinfo.value.args[0] == 'numerator and denominator must be integers'

def test_fraction_str(set_up_fractions):
    """Test various cases for specs.Fraction.__str__."""
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"

    # Test that Fraction with denom 1 only prints numer.
    assert str(frac_2_1) == '2', '2/1 -> 2'

def test_fraction_float(set_up_fractions):
    """Test various cases for specs.Fraction.__float__."""
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == 0.5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    """Test various cases for specs.Fraction.__init__."""
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions

    # Compare with Fractions.
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)

    # Compare with non-Fractions.
    assert frac_2_1 == 2, 'compare with int'
    assert frac_2_1 == 2.0, 'compare with float'

def test_fraction_add(set_up_fractions):
    """Test various cases for specs.Fraction.__init__."""
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions

    # Test adding Fractions.
    assert frac_1_2 + frac_1_3 == specs.Fraction(5, 6), '1/2 + 1/3 -> 5/6'
    assert frac_1_3 + frac_1_2 == specs.Fraction(5, 6), '1/3 + 1/2 -> 5/6'
    assert frac_n2_3 + frac_1_3 == specs.Fraction(-1, 3), '-2/3 + 1/3 -> -1/3'

def test_fraction_sub(set_up_fractions):
    """Test various cases for specs.Fraction.__sub__."""
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions

    # Test subtracting Fractions.
    assert frac_1_2 - frac_1_3 == specs.Fraction(1, 6), '1/2 - 1/3 -> 1/6'
    assert frac_2_1 - frac_1_2 == specs.Fraction(3, 2), '2 - 1/2 -> 3/2'
    assert frac_2_1 - frac_n2_3 == specs.Fraction(8, 3), '2 - -2/3 -> 8/3'

def test_fraction_mul(set_up_fractions):
    """Test various cases for specs.Fraction.__mul__."""
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions

    # Test multiplying Fractions.
    assert frac_1_2 * frac_1_3 == specs.Fraction(1, 6), '1/2 * 1/3 -> 1/6'
    assert frac_2_1 * frac_1_2 == specs.Fraction(1, 1), '2 * 1/2 -> 1/1'
    assert frac_2_1 * frac_n2_3 == specs.Fraction(-4, 3), '2 * -2/3 -> -4/3'

def test_fraction_truediv(set_up_fractions):
    """Test various cases for specs.Fraction.__truediv__."""
    frac_1_3, frac_1_2, frac_n2_3, frac_2_1 = set_up_fractions

    # Test dividing Fractions.
    assert frac_1_2 / frac_1_3 == specs.Fraction(3, 2), '1/2 / 1/3 -> 3/2'
    assert frac_2_1 / frac_1_2 == specs.Fraction(4, 1), '2 / 1/2 -> 4/1'
    assert frac_2_1 / frac_n2_3 == specs.Fraction(-3, 1), '2 / -2/3 -> -3/1'

    # Test ZeroDivisionError.
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_0_1 = specs.Fraction(0, 1)
        frac_1_2 / frac_0_1
        assert excinfo.value.args[0] == 'cannot divide by zero'


# Problem 5: Write test cases for Set.
def test_count_sets():
    """Test for specs_count_sets."""

    # Assign the function to a shorter variable name.
    t = specs.count_sets

    hand1 = ['1022', '1122', '0100', '2021',
             '0010', '2201', '2111', '0020',
             '1102', '0210', '2110', '1020']
             
    assert t(hand1) == 6, 'hand with 6 sets'

def test_is_set():
    """Test various cases for specs_count_sets."""

    # Assign the function to a shorter variable name.
    t = specs.is_set

    assert t('0000', '1111', '2222') == True, 'all different'
    assert t('0120', '1201', '2012') == True, 'all different'
    assert t('0120', '0121', '0122') == True, 'all same first 3 props, all different last prop'
    assert t('0120', '1201', '0000') == False, 'not all different'
