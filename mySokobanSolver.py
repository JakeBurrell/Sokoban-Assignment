
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

from cv2 import transform
from grpc import StatusCode
import search 
import sokoban

from sokoban import Warehouse
from itertools import permutations
import numpy as np
import re




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
    return [ (9712291, 'Jake', 'Burrell') ]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



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
    corners = set()
    interior = get_warehouse_interior(warehouse)
    # Check rule 1 on all interior spaces
    for position in interior:
        if (SokobanPuzzle.is_corner(position, warehouse.walls) and position not in warehouse.targets):
            corners.add(position)


    taboo = set()

    # Check rule 2
    # Check each corner not already in taboo
    for corner in corners:
        
        taboo.add(corner)
        # Get interior cells on the same row as corner
        row_cells = [cell for cell in interior if cell[0] == corner[0]]

        taboo_edge = check_edge(row_cells, warehouse, 0)

        # Appends all edge cells to taboo if found by check edges
        if taboo_edge != None:
            taboo.update(taboo_edge)

        # Get interior cells on the same column as corner
        col_cells = [cell for cell in interior if cell[1] == corner[1]]

        taboo_edge = check_edge(col_cells, warehouse, 1)

        # Appends all edge cells to taboo if found by check edges
        if taboo_edge != None:
            taboo.update(taboo_edge)

    # Returns warehouse as a string including taboo cells

    # Get warehouse as string
    warehouse_str = str(warehouse)
    # Remove worker, target and boxes
    warehouse_str = re.sub("[\*@$!.]", " ", warehouse_str)
    
    row = 0
    column = 0
    taboo_str = ""
    for char in warehouse_str:
        pos = tuple([column, row])
        if pos in taboo:
            taboo_str += "X"
        else:
            taboo_str += char
        if char == '\n':
            row += 1
            column = 0
        else:
            column += 1

    return taboo_str

        


def check_edge(cells, warehouse, row_col):
    '''
    Checks a particular set of cells corresponding to a row or column to see if
    they reside on a taboo edge
    '''
    assert row_col == 0 or row_col == 1

    adjacent_walls = True
    has_target = False

    # Keep track of edge cells
    edge_cells = set()
    for cell in cells:

        # Check cell is not target
        if cell in warehouse.targets:
            has_target = True
            break

        # If cells are column else if row
        if (row_col == 1):
            above_cell = tuple(cell + np.array([0,1]))
            bellow_cell = tuple( cell + np.array([0,-1]))
        else:
            # Above and bellow are actually right and left
            above_cell = tuple(cell + np.array([1,0]))
            bellow_cell = tuple( cell + np.array([-1,0]))

        # Check if each cell on row or column has adjacent wall above or bellow or right or left
        if above_cell not in warehouse.walls and bellow_cell not in warehouse.walls:
            adjacent_walls = False
            break
    
        edge_cells.add(cell)

    # Returns if edges if they all have adjacent walls and no target and thus are taboo edges
    if adjacent_walls and not has_target:
        return edge_cells
    else:
        return None
        


def get_warehouse_interior(warehouse: Warehouse):
    '''
    Determines the space that makes up the interior of the warehouse

    Modified graph search

    @param warehouse:
        a Warehouse object with the worker inside the warehouse
    
    @return
        a sequence of (x,y) pairs, positions within warehouse
    '''

    def has_wall(position, walls):
        if position in walls:
            return True
        return False
 

    frontier = []
    explored = set() # initial empty set of explored states


    frontier.append(warehouse.worker)
    while frontier:
        worker = frontier.pop(0)
        explored.add(worker)
        for move in actions:
            pos = State.step(worker, move)
            if not has_wall(pos, warehouse.walls):

                if (pos not in explored and worker not in frontier):
                    frontier.append(pos)

    return list(explored)
            

actions = ['up', 'down', 'left', 'right']

class State:
    '''
    Represents the state of the problem in the form of the position of the worker
    and the position of the boxes
    '''

    def __init__(self, worker, boxes):
        self.worker = worker
        self.boxes = boxes


    def step(position, direction):
        
        assert direction in actions

        new_pos = np.array(position)
        
        if direction == 'up':
            new_pos[1] -= 1
        elif direction == "down":
            new_pos[1] += 1
        elif direction == "right":
            new_pos[0] += 1
        elif direction == "left":
            new_pos[0] -= 1

        return tuple(new_pos)

    def movement(self, action: str):
        '''
        Changes workers position in particular direction
        '''
        self.worker = State.step(self.worker, action)


            



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

    def __init__(self, nrows, ncols, initial_state: State, walls, goals, weights):
        '''
        Initializes the the sokoban puzzle

        @param nrows: the number of rows in the puzzle

        @param ncols: the number of columns in puzzle

        @param worker: the position of the worker as a tuple

        @param boxes: the position of the boxes in the puzzle as a list of tuples

        @param goals: The position of the targets in the puzzle as a list of tuples

        @param walls: The position of the walls in the puzzle as a list of tuples
#
#
        @param weights: The weights of each of the boxes
        '''

        assert isinstance(initial_state, State)

        self.nrows = nrows
        self.ncols = ncols
        self.initial = initial_state
        self.goals = goals
        self.walls = walls
        self.weights = weights

    def box_moveable(self, box, state, action):
        '''
        Checks to see if a given box is movable in a particular state
        '''

        assert isinstance(state, State)

        new_box_pos = State.step(box, action)
        if new_box_pos not in self.walls and new_box_pos not in state.boxes:
            return True
        else:
            return False

    
    def result(self, state : State, action):
        '''
        Returns the change in state from particular action
        '''

        assert isinstance(state, State)

        new_state = state
        new_worker_pos = State.step(state.worker, action)
        if new_worker_pos in state.boxes:
            box_num = state.boxes.index(new_worker_pos)
            new_state.worker = new_worker_pos
            new_state.boxes[box_num] = State.step(state.boxes[box_num], action)


    def actions(self, state: State):
        """
        Return the list of actions that can be executed in the given state.
        
        In this case it will be the direction the worker can move
        
        """
        assert isinstance(state, State)

        possible_actions = []
        for move in actions:
            new_pos = State.step(state.worker, move)
            # Checks new position not in walls
            if new_pos not in self.walls:
                # Checks to see if new pos moves box
                if new_pos in state.boxes:
                    box_num = state.boxes.index(new_pos)
                    # If moves box check box is movable
                    new_box_pos = State.step(state.boxes[box_num], move)
                    self.box_moveable(new_box_pos, state, move)

                possible_actions.append(move)
            new_pos = state.worker
        return possible_actions

    
    def is_corner(position: tuple, walls):
        # Transforms to determine adjacent positions
        transforms = list([(-1, 0), (0,-1), (1, 0), (0, 1)])
        # Remove diagonal tranforms and convert to numpy array
        transforms = np.array([i for i in transforms if 0 in i])
        # Determine Adjacent cell
        adjacent_positions = list(map(tuple, [t + [position[0], position[1]] for t in transforms]))
        adjacent_positions.append(adjacent_positions[0])
        for index in range(len(adjacent_positions)-1):
            if adjacent_positions[index] in walls and adjacent_positions[(index + 1)] in walls:
                return True
        return False


        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequentransform = list(permutations([0,1,-1], 2)) actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Wdef test_taboo_cells():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = '####  \n#X #  \n#  ###\n#   X#\n#   X#\n#XX###\n####  '
    answer = taboo_cells(wh)
    fcn = test_taboo_cells    
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)arehouse object

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
        
        If a solution was found, transform = list(
if __name__ == "__main__":
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_03.txt")ight', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
            C is the total cost of the action sequence C

    '''
    
    raise NotImplementedError()






# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# - - - - - - - - - - - - - - - - - - -Tests- - - - - - - - - - - - - - - - - - 

def test_movement():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = tuple((4,3))
    state = State((4,4), (2,2))
    state.movement("up")
    answer = state.worker
    fcn = state.movement 
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)



def test_get_warehouse_interior():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = [(4, 4), (2, 4), (1, 2), (2, 1), (3, 4), (4, 3), (1, 5), (1, 1), (1, 4), (2, 3), (3, 3), (2, 2), (2, 5), (1, 3)]
    answer = get_warehouse_interior(wh)
    fcn = get_warehouse_interior
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)


def test_is_corner():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = [(4, 4), (2, 1), (4, 3), (1, 5), (1, 1), (2, 5)]
    answers = []
    
    interior_cells = [(4, 4), (2, 4), (1, 2), (2, 1), (3, 4), (4, 3), (1, 5), (1, 1), (1, 4), (2, 3), (3, 3), (2, 2), (2, 5), (1, 3)]
    for interior in interior_cells:
        res = SokobanPuzzle.is_corner(interior, wh.walls)
        if res:
            answers.append(interior)

    answer = answers
    fcn = SokobanPuzzle.is_corner    
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)

def test_check_edge():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = {(5, 1), (5, 2)}
    cells = tuple([(5,1),(5,2)])
    answer = check_edge(cells, wh, 1)
    fcn = check_edge 
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)

    
def test_taboo_cells():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = '####  \n#X #  \n#  ###\n#   X#\n#   X#\n#XX###\n####  '
    answer = taboo_cells(wh)
    fcn = test_taboo_cells    
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)



if __name__ == '__main__':
    print("\n\n------------------Tests----------------- \n\n")
    test_movement()
    test_get_warehouse_interior()
    test_is_corner()
    test_check_edge()
    test_taboo_cells()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# - - - - - - - - - - - - - - - - -Code Cemetary- - - - - - - - - - - - - - - - 

#class Cell:
#    '''
#    A cell within the warhouse
#    '''
#
#    MOVES = ['up', 'down', 'left', 'right']
#
#    def __init__(self, pos: tuple):
#        assert isinstance(pos, tuple)
#        self.x = pos[0]
#        self.y = pos[1]
#
#    def coordinates(self):
#        return tuple((self.x, self.y))
#
#
#
#    def is_corner(self, warehouse: Warehouse):
#        # Generate transforms to determine adjacent positions
#        transform = list(permutations([0,1,-1], 2))
#        # Remove diagonal tranforms and convert to numpy array
#        transforms = np.array([i for i in transform if 0 in i])
#        # Determine Adjacent cell
#        adjacent_positions = list(map(tuple, [t + [self.x, self.y] for t in transforms]))
#
#        for position in adjacent_positions:
#            if position in warehouse.walls and tuple(reversed(position)) in warehouse.walls:
#                return True
#        return False
#
#        
#
#
#class Path:
#    '''
#    A path within the warehouse 
#    '''
#
#    def __init__(self, position: Tuple):
#        assert len(position) == 2
#        self.steps = []
#        self.steps.append(position) 
#    
#
#class Boxes:
#    '''
#        Boxes within the warehouse
#    '''
#
#    def __init__(self, position: Tuple, weight):
#        self.position = position
#        self.weight = weight
#
#    def moves(self, warehouse: Warehouse, taboo = []):
#        '''
#        Returns a list of possible possitions that could be moved to
#        '''
#        moves = []
#        position = self.positionclass Cell:
#    '''
#    A cell within the warhouse
#    '''
#
#    MOVES = ['up', 'down', 'left', 'right']
#
#    def __init__(self, pos: tuple):
#        assert isinstance(pos, tuple)
#        self.x = pos[0]
#        self.y = pos[1]
#
#    def coordinates(self):
#        return tuple((self.x, self.y))
#
#
#
#    def is_corner(self, warehouse: Warehouse):
#        # Generate transforms to determine adjacent positions
#        transform = list(permutations([0,1,-1], 2))
#        # Remove diagonal tranforms and convert to numpy array
#        transforms = np.array([i for i in transform if 0 in i])
#        # Determine Adjacent cell
#        adjacent_positions = list(map(tuple, [t + [self.x, self.y] for t in transforms]))
#
#        for position in adjacent_positions:
#            if position in warehouse.walls and tuple(reversed(position)) in warehouse.walls:
#                return True
#        return False
#
#        
#



