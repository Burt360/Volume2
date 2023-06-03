import pytest
from binary_trees import SinglyLinkedList, BST

def test_linked_list():
    return
    a = SinglyLinkedList()

    # The list is empty.
    with pytest.raises(ValueError):
        a.iterative_find('a')
    with pytest.raises(ValueError):
        a.recursive_find('a')

    a.append('z')
    a.append('y')
    a.append('x')

    # The list is not empty.
    # Look for something in the list and something not in the list.
    assert a.iterative_find('y').value == 'y'
    with pytest.raises(ValueError):
        a.iterative_find('a')
    
    assert a.recursive_find('y').value == 'y'
    with pytest.raises(ValueError):
        a.recursive_find('a')

def test_bst():
    
    a = BST()

    # The BST is empty.
    with pytest.raises(ValueError):
        a.find(5)

    a.insert(5)

    # The BST is not empty.
    assert a.find(5).value == 5
    # Value already in the tree.
    with pytest.raises(ValueError):
        a.insert(5)

    vals = [2, 1, 7, 6, 8]
    for val in vals:
        a.insert(val)

    # Look for something in the BST and something not in the BST.
    assert a.find(8).value == 8
    with pytest.raises(ValueError):
        a.find(10)

    a.insert(3)

    """
         5
       /   \
      2     7
     / \   / \
    1   3 6   8   
    """

    # Try to remove something not in the tree.
    with pytest.raises(ValueError):
        a.remove(0)
    
    # Remove a leaf.
    a.remove(8)
    with pytest.raises(ValueError):
        a.find(8)
    
    """
         5
       /   \
      2     7
     / \   /
    1   3 6   
    """
    
    # Remove a node with one child.
    a.remove(7)
    with pytest.raises(ValueError):
        a.find(7)

    """
         5
       /   \
      2     6
     / \
    1   3  
    """
    
    # Remove a node with two children.
    a.remove(5)
    with pytest.raises(ValueError):
        a.find(5)
    
    """
         3
       /   \
      2     6
     /
    1  
    """

    a.insert(2.5)
    a.insert(2.75)

    """
         3
       /   \
      2     6
     / \
    1  2.5
         \
         2.75
    """

    a.remove(3)
    """
        2.75
       /   \
      2     6
     / \
    1  2.5
    """

    with pytest.raises(ValueError):
        a.find(3)
    assert a.root.value == 2.75

def test_bst2():
    
    a = BST()

    for num in [4] + [2, 10] + [1, 3, 5, 11] + [6, 15] + [9, 14, 16] + [7, 12]:
        a.insert(num)
    
    print(a)

    a.remove(2)

    print(a)