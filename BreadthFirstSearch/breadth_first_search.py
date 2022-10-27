# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Nathan Schill
Section 3
Thurs. Nov. 3, 2022
"""

from collections import deque
import networkx as nx
from matplotlib import pyplot as plt
import statistics as stats

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        # Add a node with an empty list of edges if the node is not alrady in the graph
        if n not in self.d:
            self.d[n] = set()

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        # Add the nodes if needed
        self.add_node(u)
        self.add_node(v)
        
        # For each node, add the edge to the other
        self.d[u].add(v)
        self.d[v].add(u)

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # Raise an error if the node is not in the graph
        if n not in self.d:
            raise KeyError('n is not in the graph')
        
        # For each neighbor of n, discard the edge from that neighbor to n
        for u in self.d[n]:
            self.d[u].discard(n)

        # Remove n
        self.d.pop(n)

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        # Raise an error if u or v are not in the graph or if there is no edge between them
        if u not in self.d:
            raise KeyError('u is not in the graph')
        if v not in self.d:
            raise KeyError('v is not in the graph')
        if v not in self.d[u]:
            # Should not need to check if u is in self.d[v] since graph is undirected
            raise KeyError('there is not an edge between u and v')
        
        # Remove both nodes
        self.d[u].remove(v)
        self.d[v].remove(u)

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        
        # Raise a KeyError if the source node is not in the graph
        if source not in self.d:
            raise KeyError('source node is not in the graph')
        
        # Init data structures for BFS with source added to Q and M
        V = list()
        Q = deque(source)
        M = set(source)

        # While Q is not empty
        while Q:
            # Pop the next node in Q and "visit" it by appending it to V
            current = Q.popleft()
            V.append(current)

            # For each neighbor u of current not already in M
            for u in self.d[current] - M:
                # Push u onto Q and add u to M
                Q.append(u)
                M.add(u)
        
        # Return the list of nodes in visitation order
        return V
    

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        
        # Raise a KeyError if either input node is not in the graph
        if source not in self.d:
            raise KeyError('source node is not in the graph')
        if target not in self.d:
            raise KeyError('target node is not in the graph')
        
        # Init data structures for BFS with source added to Q and M
        Q = deque(source)
        M = {source : None}

        # While Q is not empty
        while Q:
            # Pop the next node in Q
            current = Q.popleft()

            # For each neighbor u of current not already in M
            for u in self.d[current] - set(M):
                # Push u onto Q
                Q.append(u)

                # Add u to M mapping to the visiting node current
                M[u] = current
            
            # If the target has been found, break
            if target in M:
                break
        
        # Init a deque to store the shortest path from source to target, and x as the target
        V = deque()
        x = target
        # While x is not None, it is the next node in the reversed shortest path from target to source,
        # so appendleft it to V and set x as the next node in the path
        while x is not None:
            V.appendleft(x)
            x = M[x]

        # Return the list of nodes in the shortest path from source to target
        return list(V)


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        
        # Init Graph and sets for movie titles and actor names
        self.G = nx.Graph()
        self.movie_titles = set()
        self.actor_names = set()

        # Open the file
        with open(filename, 'r', encoding='utf8') as file:
            # Read each line and split it by '/'
            for line in file:
                L = line.split('/')

                # The first string is the movie
                self.movie_titles.add(L[0].strip())

                # The remaining strings are actors
                # Add an edge between the movie and each actor
                for actor_name in L[1:]:
                    self.actor_names.add(actor_name.strip())
                    self.G.add_edge(L[0], actor_name.strip())
                

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        # Get the shortest path between source and target
        path = nx.shortest_path(self.G, source, target)

        # The degree of separation is (path's length - 1)/2
        # (E.g., path length 5 contains 3 actors and 2 movies, so the degree is 2)
        deg = int((len(path)-1)/2)

        return path, deg 

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        
        # Get the path lengths
        path_lengths = nx.shortest_path_length(self.G, target)

        # Get the degrees of separation for each actor
        degrees = [int((length)/2) for node, length in path_lengths.items() if node not in self.movie_titles]

        # Plot the histogram
        plt.hist(degrees, bins=[i-.5 for i in range(8)])
        plt.show()
        
        # Return the average degrees of separation
        return stats.mean(degrees)