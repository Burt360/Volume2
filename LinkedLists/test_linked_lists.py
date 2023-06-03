from linked_lists import *
import pytest

@pytest.fixture
def set_up():
    return

def test_Node():
    for object in (list(), set(), dict(), len):
        with pytest.raises(TypeError) as excinfo:
            node = Node(object)
        assert excinfo.value.args[0] == 'type must be int, float, or str'

def test_LinkedList():
    ll = LinkedList()

    with pytest.raises(ValueError) as excinfo:
        ll.find(2)
    assert excinfo.value.args[0] == 'LinkedList is empty'

    [ll.append(data) for data in (2, 4, 6)]

    with pytest.raises(ValueError) as excinfo:
        ll.find(7)
    assert excinfo.value.args[0] == 'data not found'

    assert ll.find(2).value == 2, "didn't find 2"
    
    with pytest.raises(IndexError) as excinfo:
        ll.get(-1)
    assert excinfo.value.args[0] == 'index should be positive and strictly less than the length of the list'

    with pytest.raises(IndexError) as excinfo:
        ll.get(3)
    assert excinfo.value.args[0] == 'index should be positive and strictly less than the length of the list'

    assert ll.get(2).value == 6, "didn't find correct Node at index 2"


    assert len(ll) == 3, 'length not 3'

    assert str(ll) == "[2, 4, 6]"

    llstr = LinkedList()
    [llstr.append(data) for data in ('2', '4', '6', "don't")]
    
    assert str(llstr) == """['2', '4', '6', "don't"]"""


    with pytest.raises(ValueError) as excinfo:
        ll.remove(7)
    assert excinfo.value.args[0] == 'data not found'

    ll.append(8)
    ll.append(10)
    ll.remove(2)
    assert str(ll) == "[4, 6, 8, 10]"
    ll.remove(10)
    assert str(ll) == "[4, 6, 8]"
    ll.remove(6)
    assert str(ll) == "[4, 8]"
    
    ll1 = LinkedList()
    ll1.append(1)
    ll1.remove(1)
    assert str(ll1) == "[]"


    with pytest.raises(ValueError) as excinfo:
        ll.insert(-1, 0)
    assert excinfo.value.args[0] == 'index should be positive and less than or equal to the length of the list'
    with pytest.raises(ValueError) as excinfo:
        ll.insert(3, 100)
    assert excinfo.value.args[0] == 'index should be positive and less than or equal to the length of the list'

    ll.insert(1, 6)
    assert str(ll) == "[4, 6, 8]"
    ll.insert(0, 2)
    assert str(ll) == "[2, 4, 6, 8]"
    ll.insert(4, 10)
    assert str(ll) == "[2, 4, 6, 8, 10]"


def test_Deque():
    dd = Deque()

    assert str(dd) == "[]"
    
    with pytest.raises(ValueError) as excinfo:
        dd.pop()
    assert excinfo.value.args[0] == 'deque is empty'

    with pytest.raises(ValueError) as excinfo:
        dd.popleft()
    assert excinfo.value.args[0] == 'deque is empty'

    dd.append(2)
    dd.append(4)
    assert str(dd) == '[2, 4]'

    assert dd.pop() == 4
    assert str(dd) == '[2]'
    assert dd.pop() == 2
    assert str(dd) == '[]'

    dd.appendleft(2)
    assert str(dd) == '[2]'

    dd.appendleft(0)
    dd.append(4)
    assert str(dd) == '[0, 2, 4]'

    assert dd.popleft() == 0
    assert str(dd) == '[2, 4]'

    assert dd.popleft() == 2
    assert dd.popleft() == 4
    assert str(dd) == '[]'

    with pytest.raises(NotImplementedError) as excinfo:
        dd.remove()
    assert excinfo.value.args[0] == 'use pop() or popleft() for removal'

    with pytest.raises(NotImplementedError) as excinfo:
        dd.insert()
    assert excinfo.value.args[0] == 'use append() or appendleft() for removal'