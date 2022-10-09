# binary_trees.py
"""Volume 2: Binary Trees.
Nathan Schill
Sec. 3
Thurs. Oct. 13, 2022
"""

import random as rd
from time import perf_counter as pc

# These imports are used in BST.draw().
from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        
        def _step(self, data, node):
            # Checks the given node.
            
            # If the node is None, the end of the list has been reached
            # (either the list is empty or the value is not in the list).
            if node is None:
                raise ValueError(str(data) + " is not in the list")
            
            # If the node contains the value, return it.
            elif node.value == data:
                return node
            
            # Otherwise, check the next node.
            else:
                return _step(self, data, node.next)
            
        # Start the recursive search from the list's head.
        return _step(self, data, self.head)


class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        
        # If the tree is empty, insert the node at the root.
        if self.root is None:
            self.root = BSTNode(data)
            return self.root

        # Otherwise, define a recursive function to traverse the tree.
        def _step(current, parent=None):
            """Recursively step through the tree until the location where the node
            containing the data is found. If there is already a node there, raise a Value Error.
            """
            
            # Base case 1: found where the node should be inserted.
            if current is None:
                # If the data is less than the parent's value, double link on the left of parent.
                if data < parent.value:
                    parent.left = BSTNode(data)
                    parent.left.prev = parent
                    return parent.left
                
                # If the data is greater than the parent's value, double link on the right of parent.
                else:
                    parent.right = BSTNode(data)
                    parent.right.prev = parent
                    return parent.right

            # Base case 2: data already in tree.
            if data == current.value:
                raise ValueError(str(data) + " is already in the tree.")

            # Recursively search left.
            if data < current.value:
                return _step(current.left, current)
            
            # Recursively search right.
            else:
                return _step(current.right, current)

        # Start the recursion on the root of the tree.
        return _step(self.root)
        

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        
        # Find the target. If the tree is empty or the value is not in the tree, find will raise a ValueError.
        target = self.find(data)

        # If the target is a leaf node:
        if target.left is None and target.right is None:
            # If the target is the root:
            if self.root == target:
                # Remove the only access to the target.
                self.root = None
            # If the target is to the left of its parent:
            elif target.value < target.prev.value:
                # Remove the parent's left link.
                target.prev.left = None
            # If the target is to the right of its parent:
            else:
                # Remove the parent's right link.
                target.prev.right = None
        
        # If the target has two children:
        elif target.left is not None and target.right is not None:
            # Find the inorder predecessor.
            pred = target.left
            while pred.right is not None:
                pred = pred.right

            pred_val = pred.value
            # Remove the in-order predecessor using remove on its value.
            self.remove(pred_val)

            # Replace the value of the target with the value of the in-order predecessor.
            self.root.value = pred_val
        
        # If the target has one child (it has at least one because
        # it's not the case that both are None and
        # it's not the case that both are not None):

        # If the target has a left child: 
        elif target.left is not None:
            # Promote the left child and remove links to the target.

            # If the target is the root:
            if self.root == target:
                # Set the target's left child as the root.
                self.root = self.root.left
            
            # If the target is to the left of its parent:
            elif target.value < target.prev.value:
                # Set the target's left (only) child as the parent's left child.
                target.prev.left = target.left

                # Set the promoted child's parent as the target's old parent.
                target.left.prev = target.prev
            
            # If the target is to the right of its parent:
            else:
                # Set the target's left (only) child as the parent's right child.
                target.prev.right = target.left

                # Set the target's old parent as the promoted child's parent.
                target.left.prev = target.prev

        # If the target has a right child:
        else:
            # Promote the right child and remove links to the target.

            # If the target is the root:
            if self.root == target:
                # Set the target's right child as the root.
                self.root = self.root.right
            
            # If the target is to the left of its parent:
            elif target.value < target.prev.value:
                # Set the target's right (only) child as the parent's left child.
                target.prev.left = target.right

                # Set the promoted child's parent as the target's old parent.
                target.right.prev = target.prev
            
            # If the target is to the right of its parent:
            else:
                # Set the target's right (only) child as the parent's right child.
                target.prev.right = target.right

                # Set the target's old parent as the promoted child's parent.
                target.right.prev = target.prev

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()

def draw_test():
    # Figure 2.2
    vals = [5, 2, 1, 7, 6, 8, 3]
    a = BST()
    for val in vals:
        a.insert(val)
    a.draw()

    a.remove(8)
    a.draw()

    a.remove(7)
    a.draw()

    a.remove(5)
    a.insert(2.5)
    a.insert(2.75)
    a.remove(3)

    a.draw()


    # Figure 2.3
    vals = [5, 2, 7, 1, 3, 8]
    b = BST()
    for val in vals:
        b.insert(val)

    #b.draw()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    def run(n):
        """Inserts n items and finds 5 items for each tree,
        and appends the time to do so for each tree
        to lists in insert_times and find_times."""

        # Use the same n random items to insert and 5 to find for each tree
        insert_items = rd.sample(words, n)
        find_items = rd.sample(insert_items, 5)      

        # A dictionary for the trees
        trees = {
            'LinkedList' : SinglyLinkedList(),
            'BST' : BST(),
            'AVL' : AVL()
        }

        for name, tree in trees.items():
            # Use the correct insert/find method depending on the tree
            if name == 'LinkedList':
                add = tree.append
                find = tree.iterative_find
            else:
                add = tree.insert
                find = tree.find

            # Time inserting n items
            start = pc()
            for item in insert_items:
                add(item)
            end = pc()
            insert_times[name].append(end - start)

            # Time finding 5 items
            start = pc()
            for item in find_items:
                find(item)
            end = pc()
            find_times[name].append(end - start)
    
    # Store each word (one from each line) in a list
    FILEPATH = 'english.txt'
    with open(FILEPATH, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    
    # Value of n to time
    N = tuple(2**i for i in range(3, 10 + 1))

    # Lists of times to insert n items for each tree
    insert_times = {
            'LinkedList' : list(),
            'BST' : list(),
            'AVL' : list()
        }

    # Lists of times to find 5 items for each tree
    find_times = {
            'LinkedList' : list(),
            'BST' : list(),
            'AVL' : list()
        }

    # Get times for each n.
    for n in N:
        run(n)
    
    ### Insert times subplot
    sub = plt.subplot(121)
    for tree, times in insert_times.items():
        sub.loglog(N, times, label=tree)
    
    # Legend, labels, title
    sub.legend()
    sub.set_xlabel('n')
    sub.set_ylabel('time (seconds)')
    sub.set_title('Time to insert n random items')

    ### Find times subplot
    sub = plt.subplot(122)
    for tree, times in find_times.items():
        sub.loglog(N, times, label=tree)
    
    # Legend, labels, title
    sub.legend()
    sub.set_xlabel('n')
    sub.set_title('Time to find random 5 items')
    
    # Set x-axis base 2, title plot, and show
    plt.xscale('log', base=2)
    plt.suptitle('Comparison of linked list, BST, and AVL')
    plt.show()

prob4()
