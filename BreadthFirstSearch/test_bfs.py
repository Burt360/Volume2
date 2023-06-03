import pytest
import breadth_first_search as bfs

def test_Graph():
    adj =  {'A': {'B', 'D'},  
            'B': {'A', 'D'},
            'C': {'D'},
            'D': {'A', 'B', 'C'}}
    g = bfs.Graph(adj)

    assert set(adj.keys()) == set(g.d.keys())
    print(g)

    g.add_node('D')
    print(g)

    g.add_node('E')
    g.add_node('F')
    print(g)

    g.add_edge('E', 'F')
    print(g)

    g.add_edge('C', 'E')
    print(g)

    
    with pytest.raises(KeyError):
        g.remove_node('Z')
    
    g.remove_node('F')
    print(g)

    with pytest.raises(KeyError):
        g.remove_edge('E', 'F')
    with pytest.raises(KeyError):
        g.remove_edge('A', 'C')
    
    g.remove_edge('C', 'E')
    print(g)

def test_traverse():
    adj =  {'A': {'B', 'D'},  
            'B': {'A', 'D'},
            'C': {'D'},
            'D': {'A', 'B', 'C'}}
    g = bfs.Graph(adj)

    with pytest.raises(KeyError):
        g.traverse('E')

    print(g.traverse('A'))

    adj2 = {'A': {'B', 'C'}, 'B': {'E', 'A', 'D'}, 'C': {'F', 'A', 'G'}, 'D': {'B', 'I', 'H'}, 'E': {'J', 'B', 'K'}, 'F': {'M', 'C', 'L'}, 'G': {'O', 'C', 'N'}, 'H': {'D'}, 'I': {'D'}, 'J': {'E'}, 'K': {'E'}, 'L': {'F'}, 'M': {'F'}, 'N': {'G'}, 'O': {'G'}}
    g = bfs.Graph(adj2)
    print(g.traverse('A'))
    assert str(g.traverse('A')) == "['A', 'B', 'C', 'E', 'D', 'F', 'G', 'J', 'K', 'I', 'H', 'M', 'L', 'O', 'N']"

def test_shortest_path():
    
    adj =  {'A': {'B', 'D'},  
            'B': {'A', 'D'},
            'C': {'D'},
            'D': {'A', 'B', 'C'}}
    g = bfs.Graph(adj)

    with pytest.raises(KeyError):
        g.shortest_path('A', 'E')
    with pytest.raises(KeyError):
        g.shortest_path('E', 'A')
    
    print(g.shortest_path('A', 'C'))

def test_MovieGraph():
    return
    g = bfs.MovieGraph()

    assert len(g.movie_titles) == 137018
    assert len(g.actor_names) == 930717

    actor0 = 'Kevin Bacon'
    actor1 = 'Toby Jones'

    goal_path = ['Kevin Bacon', 'Frost|Nixon (2008)', 'Toby Jones']
    goal_degrees_sep = 1

    path, deg = g.path_to_actor(actor0, actor1)
    assert path == goal_path
    assert deg == goal_degrees_sep

    print(g.path_to_actor(actor0, 'Charles Perry'))

    print(g.average_number('Kevin Bacon'))