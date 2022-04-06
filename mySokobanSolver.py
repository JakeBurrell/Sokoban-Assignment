
'''

    Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

You are NOT allowed to change the defined interfaces.
In other words, you must fully adhere to the specifications of the 
functions, their arguments and returned values.
Changing the interfacce of a function will likely result in a fail 
for the test of your code. This is not negotiable! 

You have to make sure that your code works with the files provided 
(search.py and sokoban.py) as your code will be tested 
with the original copies of these files. 

Last modified by 2022-03-27  by f.maire@qut.edu.au
- clarifiy some comments, rename some functions
  (and hopefully didn't introduce any bug!)

'''

# You have to make sure that your code works with 
# the files provided (search.py and sokoban.py) as your code will be tested 
# with these files
from turtle import position
from typing import Tuple
import search 
import sokoban
from sokoban import Warehouse




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [ (9712291, 'Jake', 'Burrell') ]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Suggested to use
# itertools.combinations
# Adapt a breadth first search 

def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A "taboo cell" is by definition
    a cell inside a warehouse such that whenever a box get pushed on such 
    a cell then the puzzle becomes unsolvable. 
    
    Cells outside the warehouse are not taboo. It is a fail to tag an 
    outside cell as taboo.
    
    When determining the taboo cells, you must ignore all the existing boxes, 
    only consider the walls and the target  cells.  
    Use only the following rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.
    
    @param warehouse: 
        a Warehouse object with the worker inside the warehouse

    @return
       A string representing the warehouse with only the wall cells marked with 
       a '#' and the taboo cells marked with a 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()


def get_warehouse_interior(warehouse: Warehouse):
    '''
    Determines the space that make up the interior of the warehouse

    @param warehouse:
        a Warehouse object with the worker inside the warehouse
    
    @return
        a sequence of (x,y) pairs, positions within warehouse
    '''
    visited = set()
    frontier = []

    frontier.append(Worker(warehouse.worker))
    while frontier:
        worker = frontier.pop()
        visited.add(worker.position())
        for worker in worker.moves(warehouse):
            if (worker.position not in visited
                and worker not in frontier):
                frontier.append(worker)

    return visited
            
            



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 
    
    '''
    
    #
    #         "INSERT YOUR CODE HERE"
    #
    #     Revisit the sliding puzzle and the pancake puzzle for inspiration!
    #
    #     Note that you will need to add several functions to 
    #     complete this class. For example, a 'result' method is needed
    #     to satisfy the interface of 'search.Problem'.
    #
    #     You are allowed (and encouraged) to use auxiliary functions and classes

    
    def __init__(self, warehouse):
        raise NotImplementedError()

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.
        
        """
        raise NotImplementedError

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Impossible', if one of the action was not valid.
           For example, if the agent tries to push two boxes at the same time,
                        or push a box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    ##         "INSERT YOUR CODE HERE"
    
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban(warehouse):
    '''
    This function analyses the given warehouse.
    It returns the two items. The first item is an action sequence solution. 
    The second item is the total cost of this action sequence.
    
    @param 
     warehouse: a valid Warehouse object

    @return
    
        If puzzle cannot be solved 
            return 'Impossible', None
        
        If a solution was found, 
            return S, C 
            where S is a list of actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
            C is the total cost of the action sequence C

    '''
    
    raise NotImplementedError()




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - -Classes- - - - - - - - - - - - - - - - - 


MOVES = ['up', 'down', 'left', 'right']


class Worker:
    '''
    A worker within the warehouse
    '''

    def __init__(self, position):

        self.x = position[0]
        self.y = position[1]

    def step(self, direction: str):
        '''
        Steps forward in the direction provided
        '''
        assert direction in MOVES
        
        if direction == 'up':
            self.y += 1
        elif direction == "down":
            self.y -= 1
        elif direction == "right":
            self.x += 1
        elif direction == "left":
            self.x -= 1
    
    def moves(self, warehouse: Warehouse):
        '''
        Returns a list of possible possitions that could be moved to
        '''
        moves = []
        position = self.position
        for move in MOVES:
            self.step(move)
            if (self.position() not in warehouse.walls):
                moves.append(self.position())
            self = Worker(position)
        return (position)

    def position(self):
        return (self.x,self.y)
  

class Path:
    '''
    A path within the warehouse 
    '''

    def __init__(self, position: Tuple):
        assert len(position) == 2
        self.steps = []
        self.steps.append(position) 
    

class Boxes:
    '''
        Boxes within the warehouse
    '''

    def __init__(self, position: Tuple, weight):
        self.position = position
        self.weight = weight

    def moves(self, warehouse: Warehouse, taboo = []):
        '''
        Returns a list of possible possitions that could be moved to
        '''
        moves = []
        position = self.position
        for move in MOVES:
            self.step(move)
            if (self.position() not in warehouse.walls
                and self.position() not in taboo):
                moves.append(self.position())
            self = Worker(position)
        return (position)

# Tests
if __name__ == '__main__':
    print("----Tests---- ")
