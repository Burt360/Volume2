def max_unimodal(seq, start=0, end=-1):
    """Returns the maximal element in unimodal seq (list)."""

    if end == -1:
        end = len(seq)-1
    
    mid = (start + end)//2  # Rounds down, so a sublist of just the first two elements
                            # of the original list may not have an element to the left of mid.
    
    #print(seq[start:end+1])

    # If the sublist has one element, then that's the maximal element.
    if start == end:
        return seq[mid]
    
    # If the element to the right of mid is greater, search the right half.
    elif seq[mid] < seq[mid + 1]:
        return max_unimodal(seq, mid+1, end)

    # If mid is 0 (and the element to the right of mid is not greater),
    # then the element at mid is the greatest.
    # (This prevents mid-1 from being -1 which results in comparing with the last element in the list.)
    elif mid == 0:
        return seq[mid]

    # If the element to the left of mid is greater, search the left half.
    elif seq[mid-1] > seq[mid]:
        return max_unimodal(seq, start, mid-1)

    # Otherwise, the element at mid is the greatest.
    else:
        return seq[mid]

def test_max_unimodal(test):
    print(max_unimodal(test))