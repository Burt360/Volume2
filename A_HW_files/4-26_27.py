### 4.26
def knapsack(W, Items, solns={0 : (0, None)}):
    """Return the max value for the {0, 1}-knapsack problem.
    
    Parameters:
        W (int): max weight
        Items (list): tuples representing items (weight [float], value [float])
        vals (dict): {weight : (value, list of ints representing how many of each item in Items (in order))}
    
    Return:
        v (float): max value achieved
        path (list): list of ints representing how many of each item in Items (in order)
    """
    if solns[0][1] is None:
        solns={0 : (0, [0] * len(Items))}
    
    # (item index, weight, value) for each item that doesn't go over the weight limit
    valid_items = [(i, item[0], item[1]) for i, item in enumerate(Items) if item[0] <= W]
    
    # List of possible solns to check which is best later
    possible_solns = list()

    # Check the best value that can be achieved using each item
    for item in valid_items:
        item_index = item[0]
        w_to_add = item[1]
        w_remaining = W - w_to_add

        # If the soln has been computed for the remaining weight:
        if w_remaining in solns:
            current_soln = solns[w_remaining]
            new_path = current_soln[1].copy()

            # If this soln doesn't already have 1 of the given item:
            if current_soln[1][item_index] == 0:
                # Store the solution with this item added
                new_path[item_index] = 1
                possible_solns.append((item[2] + current_soln[0], new_path))
            
            else:
                # Otherwise store the soln without this item added
                possible_solns.append((current_soln[0], new_path))
        
        # If the soln hasn't been computed for the remaining weight:
        else:
            # Compute the soln
            value, path = knapsack(w_remaining, Items, solns)
            new_path = path.copy()

            # If this soln doesn't already have 1 of the given item:
            if path[item_index] == 0 :
                # Store the solution with this item added
                new_path[item_index] = 1
                possible_solns.append((item[2] + value, new_path))
            else:
                # Otherwise store the soln without this item added
                possible_solns.append((value, new_path))
    
    # Save the best soln
    best_soln = solns[0]
    for possible_soln in possible_solns:
        if possible_soln[0] > best_soln[0]:
            best_soln = possible_soln
    solns[W] = best_soln

    return best_soln