# linked_lists.py
"""Volume 2: Linked Lists.
Nathan Schill
Section 3
Thurs. Oct. 6, 2022
"""

import pdb

# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
        Data must be type int, float, or str.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """

        if type(data) not in (int, float, str):
            # Check the data type and raise an error if it's not valid.
            raise TypeError('type must be int, float, or str')
        
        # Store the data in the Node.
        self.value = data


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.length = 0

        self.head = None
        self.tail = None

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
        
        # Increment the length.
        self.length += 1

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        
        if self.length == 0:
            # If the list is empty, raise an error.
            raise ValueError('LinkedList is empty')
        
        current_node = self.head
        
        # Iterate through the Nodes.
        # If the current Node's value is data, return it.
        # Otherwise, move to the next Node.
        while current_node is not None:
            if current_node.value == data:
                return current_node
            
            current_node = current_node.next
        
        if current_node is None:
            # If the loop ends and the current Node is None,
            # then the data was not in the list.
            raise ValueError('data not found')


    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        
        if i < 0 or i >= self.length:
            # If i is negative or greater than or equal to the size of the list,
            # raise an IndexError.
            raise IndexError('index should be positive and strictly less than the length of the list')

        # Iterate through i nodes, then return the Node found.
        current_node = self.head
        for _ in range(i):
            current_node = current_node.next
        return current_node

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        # Return the length of the list.
        return self.length

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """

        # Start the string with '['.
        s = '['
        if self.length != 0:
            current_node = self.head
            while current_node.next is not None:
                # If the current Node is not the last, append its value and the comma.
                s += repr(current_node.value) + ', '
                current_node = current_node.next
            # When the current Node is the last, don't append a comma.
            s += repr(current_node.value)
        # Append a ']'.
        s += ']'

        return s


    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        
        # Find the Node containing the data.
        # If no such Node exists, find() will raise a ValueError.
        node_to_remove = self.find(data)

        if node_to_remove is self.head and node_to_remove is self.tail:
            # If the Node is both the head and the tail (i.e., there is only one Node in the list),
            # set head and tail to None.
            self.head = None
            self.tail = None
        elif node_to_remove is self.head:
            # If the Node is the head, point head to the next Node
            # and point that Node's prev to None.
            self.head = node_to_remove.next
            self.head.prev = None
        elif node_to_remove is self.tail:
            # If the Node is the tail, point tail to the previous Node
            # and point that Node's next to None.
            self.tail = node_to_remove.prev
            self.tail.next = None
        else:
            # If the Node is not the head or the tail,
            # repoint the previous Node to next and the next Node to prev.
            node_to_remove.prev.next = node_to_remove.next
            node_to_remove.next.prev = node_to_remove.prev
        
        # Decrement the length of the list.
        self.length -= 1
        

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """

        if index < 0 or index > self.length:
            # If the index is negative or greater than the length of the list,
            # raise a IndexError.
            raise IndexError('index should be positive and less than or equal to the length of the list')
        
        if index == self.length:
            # If the index is the length of the list, append the new node.
            self.append(data)
        elif index == 0:
            # If the index is 0 (but the length is not zero given the previous if block),
            # point the new Node's next to the current head,
            # point the current head's prev to the new Node,
            # and set head as the new Node.
            new_node = Node(data)
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
            self.length += 1
        else:
            # Create the new Node and get the next Node.
            new_node = Node(data)
            next_node = self.get(index)

            # Point previous Node's next to the new Node
            # and the new Node's prev to the previous Node.
            next_node.prev.next = new_node
            new_node.prev = next_node.prev

            # Point the next Node's prev to the new Node
            # and the new Node's next to the next Node.
            next_node.prev = new_node
            new_node.next = next_node
            self.length += 1


# Problem 6: Deque class.
class Deque(LinkedList):
    """Deque data structure class.
    
    Implements pop, popleft, append, appendleft using LinkedList,
    and disables non-deque methods.
    """
    
    def __init__(self):
        LinkedList.__init__(self)
    
    def appendleft(self, data):
        # Call the inherited insert function at index 0.
        LinkedList.insert(self, 0, data)

    def pop(self):
        if self.length == 0:
            raise ValueError('deque is empty')
        
        if self.length == 1:
            # If there is only one item, set head and tail to None.
            node_to_remove = self.head
            self.head = None
            self.tail = None
        else:
            # Get last Node, set tail to the prior Node, and set the new tail's next to None.
            node_to_remove = self.tail
            self.tail = node_to_remove.prev
            self.tail.next = None
        
        # Decrement the length and return the popped Node's value.
        self.length -= 1
        return node_to_remove.value

    def popleft(self):
        if self.length == 0:
            raise ValueError('deque is empty')
        
        # Get the value of the first Node, then call the inherited remove function on it
        # (since remove() removes the first Node containing the given data). Return the value of the popped Node.
        data = self.head.value
        LinkedList.remove(self, data)
        return data
    
    def remove(*args, **kwargs):
        # Make remove inaccessible to users of Deque.
        raise NotImplementedError('use pop() or popleft() for removal')
    def insert(*args, **kwargs):
        # Make insert inaccessible to users of Deque.
        raise NotImplementedError('use append() or appendleft() for removal')
        
    

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    
    # Create a stack.
    stack = Deque()

    # Push each line from infile onto the stack.
    with open(infile, 'r') as file:
        for line in file.readlines():
            # Remove leading/trailing whitespace (particularly newlines).
            stack.appendleft(line.strip())
    
    # Pop each line from the stack into outfile.
    with open(outfile, 'w') as file:
        first_line = True
        while True:
            try:
                # Get the next line. If there are no more lines on the stack,
                # catch the ValueError and break from the while loop.
                line = stack.popleft()

                # Write a newline before each line except the first.
                if first_line:
                    first_line = False
                else:
                    file.write('\n')
                
                # Write each line.
                file.write(line)
            except ValueError:
                # The stack is emtpy.
                break