# regular_expressions.py
"""Volume 3: Regular Expressions.
Nathan Schill
Section 2
Thurs. Feb. 23, 2023
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # Return pattern object of 'python''
    return re.compile(r'python')


# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # Return pattern object of expr
    expr = r'\^\{\@\}\(\?\)\[\%\]\{\.\}\(\*\)\[\_\]\{\&\}\$'
    return re.compile(expr)


# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # Return pattern object of expr
    expr = r'^(Book|Mattress|Grocery) (store|supplier)$'
    return re.compile(expr)


# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    # Valid python identifier pattern
    pi = r'[_a-zA-z][_\w]*'

    # Real number pattern
    rn = r'\d+(\.\d+)'

    # Single-quote pattern
    qu = r"'[^']*'"

    # Complete pattern
    expr = fr"^{pi} *(= *{rn}|= *{qu}|= *{pi})?$"
    return re.compile(expr)


# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """

    # Keywords
    keywords = 'if', 'elif', 'for', 'while', 'with', 'def', 'class', 'except', 'else', 'finally', 'try'

    # Join keywords with pipes
    keywords = r'|'.join(keywords)
    
    # Add colons for keywords
    pattern = re.compile(fr'^(\s*?({keywords}).*)$', re.MULTILINE)
    code = pattern.sub(r'\1:', code)

    return code


# Problem 6
def prob6(filename='fake_contacts.txt'):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    
    # Create contacts dictionary, read file to string
    contacts = dict()
    with open(filename) as file:
        s = file.read()

    # Get names and store init entry in contacts dictionary
    name_expr = r'^([\.a-zA-Z ]*) '
    name_pat = re.compile(name_expr, re.MULTILINE)
    names = name_pat.findall(s)
    for name in names:
        contacts[name] = {'birthday':None, 'email':None, 'phone':None}

    # Get names and emails, and save emails
    email_expr = name_expr + r'.*?([\.\w]+@[\.\w]+)'
    email_pat = re.compile(email_expr, re.MULTILINE)
    names_emails = email_pat.findall(s)
    for name, email in names_emails:
        contacts[name]['email'] = email

    # Get names and phone numbers, and save phone numbers
    phone_expr = name_expr + r'.*?(\d\d\d)[\-\)]*(\d\d\d)[\-\(\)]*(\d\d\d\d)'
    phone_pat = re.compile(phone_expr, re.MULTILINE)
    names_phones = phone_pat.findall(s)
    for name, phone1, phone2, phone3 in names_phones:
        contacts[name]['phone'] = f'({phone1}){phone2}-{phone3}'

    # Get names and birthdays, and save birthdays
    bday_expr = name_expr + r'.*?(\d{1,2})[/](\d{1,2})[/](\d{2,4})'
    bday_pat = re.compile(bday_expr, re.MULTILINE)
    names_bdays = bday_pat.findall(s)
    for name, m, d, y in names_bdays:
        # If year has length 2, prepend '20'
        if len(y) == 2:
            y = '20' + y
        contacts[name]['birthday'] = f'{m.zfill(2)}/{d.zfill(2)}/{y}'
    
    return contacts