
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
from unittest import result

from grpc import StatusCode
import search 
import sokoban

from sokoban import Warehouse
from itertools import permutations
import numpy as np
import re
import time




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
    
    interior = get_warehouse_interior(warehouse)
    
    taboo = calculate_taboo_cells(interior, warehouse)

    return construct_taboo_string(warehouse, taboo)


def calculate_taboo_cells(interior, warehouse):
    '''
    Determins a the set of cells that would be considered taboo within a sokoban problem
    
    @param interior:
        The cells that make up the interior of the warehouse 
    @param warehouse:
        The warehouse for which to determine the taboo cells of
    @return
        A set containing all the taboo cells within the warehouse as tuples of (x,y)
    '''
    corners = set()
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
        if taboo_edge: taboo.update(taboo_edge)

        # Get interior cells on the same column as corner
        col_cells = [cell for cell in interior if cell[1] == corner[1]]

        taboo_edge = check_edge(col_cells, warehouse, 1)

        # Appends all edge cells to taboo if found by check edges
        if taboo_edge: taboo.update(taboo_edge)

    return taboo



def construct_taboo_string(warehouse: Warehouse, taboo: set):
    '''
    Returns warehouse as a string representing the warehouse including only
    walls and taboo cells

    @param warehouse:
        The warehouse from which to construct the taboo string from

    @param taboo:
        The sets of the the taboo cells as tuples in the form (x,y)
    
    @return:
        A string in the same form as a warehouse string but with the workers, 
        boxes and targets removed and the taboo cells added
    '''

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
        


def check_edge(cells, warehouse: Warehouse, row_col):
    '''
    Checks a particular set of cells corresponding to a row or column to see if
    they reside on a taboo edge

    @param cells:
        A list of cell in the form of tuples (x,y) representing a row column to be
        checked to see if they represent an edge
    
    @param warehouse:
        A Warehouse object used to check for edges
    
    @param row_col:
        An int of 1 if cells represents a column else 0 if they represent a row
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

        # If cells are row else if column
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

    # Returns edges if they all have adjacent walls and are not targets and thus are taboo edges
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
 

    frontier = []
    explored = set() # initial empty set of explored states

    # Modified breadth first search
    frontier.append(warehouse.worker)
    while frontier:
        worker = frontier.pop(0)
        explored.add(worker)
        for move in ACTIONS:
            pos = State.step(worker, move)
            if pos not in warehouse.walls:
                if (pos not in explored and worker not in frontier):
                    frontier.append(pos)

    return list(explored)
            


'''
Actions that can be made by a worker
'''
ACTIONS = ['Up', 'Down', 'Left', 'Right']

class State:
    '''
    Represents the state of the Sokoban problem in the form of the position of the worker
    and the position of the boxes
    '''

    def __init__(self, worker, boxes):
        '''
        Initializes a state within the sokoban problem

        @param worker: the position of the worker as a tuple in the for (x,y)
        
        @param boxes: the positions of the boxes as a list of tuples in the for (x,y)
        '''
        self.worker = worker
        self.boxes = boxes

    '''
    Overridden methods
    '''
    def __lt__(self, other):
        return self.worker < other.worker

    def __eq__(self, other: object):
        return isinstance(other, State) and self.__key() == other.__key()

    def __key(self):
        return str((self.worker, self.boxes))

    def __hash__(self):
        '''
        Used to check set membership 
        '''
        return hash(self.__key())


    def step(position, direction):

        '''
        Determins the change in position that would result from a step in a
        particular direction

        @param position:
            A position from which a step will be made in the form of tuple (x,y)

        @param direction:
            A direction or ACTION to move, an element of the ACTIONS array
        
        @return
            The new position resultant from the action as a tuple (x,y)

        '''
        assert direction in ACTIONS

        new_pos = np.array(position)
        
        if direction == ACTIONS[0]:
            new_pos[1] -= 1
        elif direction == ACTIONS[1]:
            new_pos[1] += 1
        elif direction == ACTIONS[3]:
            new_pos[0] += 1
        elif direction == ACTIONS[2]:
            new_pos[0] -= 1

        return tuple(new_pos)

    def copy(self):
        '''
        Returns a clone of this state
        '''
        clone = State(self.worker[:], self.boxes[:])
        return clone


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


    def __init__(self, nrows, ncols, initial_state: State, walls, goals, weights, taboo_cells = None):
        '''
        Initializes the the sokoban puzzle

        @param nrows: the number of rows in the puzzle

        @param ncols: the number of columns in puzzle

        @param state: the state of the particular puzzle
        
        @param walls: The position of the walls in the puzzle as a list of tuples

        @param goals: The position of the targets in the puzzle as a list of tuples
        
        @param weights: The weights of each of the boxes

        @param taboo_cells: A set of taboo cells if one wishes for them to be considered
        '''
        assert isinstance(initial_state, State)

        self.nrows = nrows
        self.ncols = ncols
        self.initial = initial_state
        self.goals = goals
        self.walls = walls
        self.weights = weights
        self.taboo_cells = taboo_cells

    def box_moveable(self, box_num, state, action):
        '''
        Checks to see if a given box is movable in a particular state in a given direction
        Note that this will check for walls and other boxes 

        @param box_num:
            The index to the particular box that is being referenced

        @param state:
            The state of the warehouse in the form of a State object containing the
            position of the worker and boxes within the warehouse

        @param action:
            An action or movement of the worker in the form of a string of ['Up', 'Down', 'Left', 'Right']

        @return:
            Returns True if the given box is movable in the given state by the given action
            else returns False
        '''


        assert isinstance(state, State)

    
        new_box_pos = State.step(state.boxes[box_num], action)
        # If new box position is not a wall and not a box
        if new_box_pos not in self.walls and new_box_pos not in state.boxes:
            # Checks if taboo cells are provided and tests if the new box position is in them
            if self.taboo_cells != None and new_box_pos in self.taboo_cells:
                return False
            return True
        return False

    
    def result(self, state : State, action):
        '''
        Determins the change in state resultant from a particular action

        @param state:
            The state of the warehouse in the form of a State object containing the
            position of the worker and boxes within the warehouse

        @param action:
            An action or movement of the worker in the form of a string of one of the
            folowing ['Up', 'Down', 'Left', 'Right']

        @return:
            The state that results from the action 
        '''


        assert isinstance(state, State)
        assert action in self.actions(state)

        new_state = state.copy()

        # Determine new worker position and change states worker position accordingly
        new_worker_pos = State.step(state.worker, action)
        new_state.worker = new_worker_pos
        # If new worker position is on a box
        if new_worker_pos in state.boxes:
            # Determine which box its on
            box_num = state.boxes.index(new_worker_pos)
            # Change states box position accordingly
            new_state.boxes[box_num] = State.step(state.boxes[box_num], action)
        return new_state


    def actions(self, state: State):
        """
        Return the list of actions that can be executed in the given state.
        
        In this case it will be the directions the worker can move

        @param state:
            The current State object from which to assess actions from
        
        @return:
            A list of possible action that could be made from the provided state in the
            form  a string of one of the folowing ['Up', 'Down', 'Left', 'Right']
        
        """
        assert isinstance(state, State)

        possible_actions = []
        # For each allowable ACTION
        for move in ACTIONS:
            new_pos = State.step(state.worker, move)
            # Checks new position not in walls
            if new_pos not in self.walls:
                # Checks to see if new pos moves box
                if new_pos in state.boxes:
                    box_num = state.boxes.index(new_pos)
                    # If moves box check box is movable
                    if self.box_moveable(box_num, state, move):
                        possible_actions.append(move)
                else:    
                    possible_actions.append(move)
            # Return new_pos to original position
            new_pos = state.worker[:]
        return possible_actions



    
    def is_corner(position: tuple, walls: list):
        
        '''
        Checks if a particular cell within a Sokoban problem is on a corner or not

        @param position:
            The position to check in the form tuple(x,y)
        
        @param walls:
            The walls of the warehouse in the form of a list of tuple(x,y)
        
        @return:
            True if and only if the position represents a corner else returns False
        '''
        assert isinstance(position, tuple or list) and isinstance(walls, list)

        # Transforms to determine adjacent cells
        transforms = np.array([(-1, 0), (0,-1), (1, 0), (0, 1)])
        # Determines Adjacent cell
        adjacent_positions = list(map(tuple, [t + [position[0], position[1]] for t in transforms]))
        # Appends first element to last element so both the first and last element can be checked together
        # I.E. the bellow element and the left element are checked if both walls together
        adjacent_positions.append(adjacent_positions[0])
        for index in range(len(adjacent_positions)-1):
            if adjacent_positions[index] in walls and adjacent_positions[(index + 1)] in walls:
                return True 
        return False

    def goal_test(self, state: State):
        """
        Return True if the state is in a goal state for this particular sokoban problem.
        """
        assert isinstance(state, State)
        for box in state.boxes:
            if box not in self.goals:
                return False
        return True


    def path_cost(self, c, state1: State, action, state2: State ):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        
        # Manhatton distance from workers state1 position to their state2 position
        worker_init = state1.worker
        worker_fin = state2.worker
        # Calculates cost incurred by worker to get fro state1 to state2
        worker_cost = manhattan_dist(worker_init, worker_fin)
        box_cost = 0
        # For each box
        for box_num, box in enumerate(state1.boxes):
            box_weight = self.weights[box_num]
            # If there is a weight incurred by the box and a movement
            if box != state2.boxes[box_num] and box_weight != 0:
                # Calculate the weight incurred by the box
                box_dist = manhattan_dist(box, state2.boxes[box_num])
                box_cost += box_dist * box_weight

        return worker_cost + box_cost + c


    def h(self, node: search.Node):
        '''
        Heuristic for goal state for the sokoban puzzle
        '''
        # Thinking this will be the average manhattan distance too all the boxes from the workers position
        # Plus the manhattan distance from each box to their closest target obviously excluding targets 
        # once assigned a box each of these values will also need to be multiplied by (1 + each box_weight)
        # note: nodes store the state

        box_distances = []
        targets = self.goals[:]
        dist_target = 0
        # For each box in the node state
        boxes = node.state.boxes[:]
        weights = self.weights[:]
        # Sort boxes such that the most weighted boxes get assigned the closest targets first
        weights, boxes = (list(t) for t in zip(*sorted(zip(weights, boxes))))
        weights.reverse()
        boxes.reverse()
        for box_num, box in enumerate(boxes):
            # Calculate manhattan distance from worker to box 
            distance_box = manhattan_dist(box, node.state.worker)
            # Store distance in box dist
            box_distances.append(distance_box)
            distance_targets = []
            # For each of the targets
            for target in targets:
                # Calculate distance and store in dist_target array
                distance = manhattan_dist(target, box)
                distance_targets.append(distance)
            # Determine which target has been assigned
            assigned_target = np.argmin(distance_targets)
            # Add the distance from box to closest target
            dist_target += (np.amin(distance_targets) * ( 1 + weights[box_num]))
            # Removes target from targets as it has already been assignment a box
            targets.pop(assigned_target)

        # Calculates the average distance to each box
        max_distance_box = np.array(box_distances).mean()
        
        total_h = max_distance_box + dist_target

        return total_h




    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""

        raise NotImplementedError


def manhattan_dist(pos_one: tuple, pos_two: tuple):
    '''
    Returns the manhattan distance between two points
    '''
    assert type(pos_one) == type(pos_two)
    assert isinstance(pos_two, tuple)

    return abs(pos_one[0] - pos_two[0]) + abs(pos_one[1] - pos_two[1])
        

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
    assert isinstance(warehouse, Warehouse)

    # Construct problem
    problem = SokobanPuzzle(warehouse.nrows, warehouse.ncols, State(warehouse.worker, warehouse.boxes),
                            warehouse.walls, warehouse.targets, warehouse.weights)
    # Constructs copy of state cause it will be returned
    current_state = problem.initial.copy()

    # For each action in the actions sequence 
    for action in action_seq:
        # If the action in the sequence is not in the actions allowed by the current state of the problem
        if action not in problem.actions(current_state):
            return 'Impossible'
        # Updates the current state with the result of the action
        current_state = problem.result(current_state, action)
    # Updates the warehouse copy with the surrent state of the worker and boxes
    warehouse.worker = current_state.worker[:]
    warehouse.boxes = current_state.boxes[:]
    return str(warehouse)
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban(warehouse: Warehouse):
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
    # Determine warehouse interior
    interior = get_warehouse_interior(warehouse)

    # Determine taboo cells
    taboo_cells = calculate_taboo_cells(interior, warehouse)

    # Construct the problem class
    problem = SokobanPuzzle(warehouse.nrows, warehouse.ncols, State(warehouse.worker, warehouse.boxes), 
    warehouse.walls, warehouse.targets, warehouse.weights, taboo_cells)

    solution_node = search.astar_graph_search(problem)

    if solution_node == None:
        return "Impossible", None
    else:
        return solution_node.solution(), solution_node.path_cost



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# - - - - - - - - - - - - - - - - - - -Tests- - - - - - - - - - - - - - - - - - 

def test_movement():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = tuple((4,3))
    state = State((4,4), (2,2))
    state.movement("Up")
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

def test_actions():
    # Test 1
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_answer = [['Up', 'Down', 'Right'], ['Right']]
    problem = SokobanPuzzle(wh.nrows, wh.ncols, State(wh.worker, wh.boxes),
     wh.walls, wh.targets, wh.weights)
    checked_state = State(wh.worker, wh.boxes)
    answer = []
    answer.append(problem.actions(checked_state))
    
    # Test 2
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_03.txt")
    problem = SokobanPuzzle(wh.nrows, wh.ncols, State(wh.worker, wh.boxes),
     wh.walls, wh.targets, wh.weights)
    checked_state = State(wh.worker, wh.boxes)
    result = problem.actions(checked_state)
    answer.append(result)

    fcn = problem.actions 
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)

def test_results():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    expected_state = State((3,4), [wh.boxes[0], (4,4)])
    expected_answer = [expected_state.worker, expected_state.boxes]
    problem = SokobanPuzzle(wh.nrows, wh.ncols, State(wh.worker, wh.boxes),
     wh.walls, wh.targets, wh.weights)
    checked_state = State((2,4), wh.boxes)
    action = ACTIONS[3]
    answer_state =  problem.result(checked_state, action )
    answer = [answer_state.worker, answer_state.boxes]
    fcn = problem.result 
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)

def test_check_elem_action_seq():
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_147.txt")
    expected_answer = "       #####\n########   #\n#*   *@  #*#\n#  ###     #\n##    #    #\n #     #####\n #  #  #    \n ## #  #    \n  #   ##    \n  #####     "
    action_seq = ['Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Down', 'Down', 'Down', 'Right', 'Right', 'Up', 'Right', 'Down', 'Right', 'Down', 'Down', 'Left', 'Down', 'Left', 'Left', 'Up', 'Up', 'Down', 'Down', 'Right', 'Right', 'Up', 'Right', 'Up', 'Up', 'Left', 'Left', 'Left', 'Down', 'Left', 'Up', 'Up', 'Up', 'Left', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Right', 'Right', 'Right', 'Up', 'Up', 'Left', 'Left', 'Down', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Down', 'Down', 'Down', 'Right', 'Right', 'Up', 'Left','Down', 'Left', 'Up', 'Up', 'Left', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Left', 'Left', 'Left', 'Left', 'Left', 'Down','Down', 'Down', 'Down', 'Right', 'Down', 'Down', 'Right', 'Right', 'Up', 'Up', 'Right', 'Up', 'Left', 'Left', 'Left', 'Down', 'Left','Up', 'Up', 'Up', 'Left', 'Up', 'Right', 'Right', 'Right', 'Right', 'Right', 'Down', 'Right', 'Down', 'Right', 'Right', 'Up', 'Left','Right', 'Right', 'Up', 'Up', 'Left', 'Left', 'Down', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Right', 'Right', 'Right', 'Right', 'Right','Right', 'Up', 'Right', 'Right', 'Down', 'Down', 'Left', 'Down', 'Left', 'Left', 'Up', 'Right', 'Right', 'Down', 'Right', 'Up', 'Left','Left', 'Up', 'Left', 'Left']
    answer = check_elem_action_seq(wh, action_seq) 
    fcn = check_elem_action_seq 
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)

def test_h():
    # Test 1
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_01.txt")
    problem = SokobanPuzzle(wh.nrows, wh.ncols, State(wh.worker, wh.boxes),
     wh.walls, wh.targets, wh.weights)
    # Actual costs [33, 396, 431] >= h  enure
    expected_answer = [7.5, 285.5, 421.0]
    state = State(wh.worker, wh.boxes)
    answer = []
    answer.append(problem.h(search.Node(state)) )
    
    # Test 2
    wh.load_warehouse("./warehouses/warehouse_09.txt")
    problem = SokobanPuzzle(wh.nrows, wh.ncols, State(wh.worker, wh.boxes),
     wh.walls, wh.targets, wh.weights)
    state = State(wh.worker, wh.boxes)
    answer.append(problem.h(search.Node(state)) )

    # Test 3
    wh.load_warehouse("./warehouses/warehouse_8a.txt")
    problem = SokobanPuzzle(wh.nrows, wh.ncols, State(wh.worker, wh.boxes),
     wh.walls, wh.targets, wh.weights)
    state = State(wh.worker, wh.boxes)
    answer.append(problem.h(search.Node(state)) )
    fcn = problem.h
    print('<<  Testing {} >>'.format(fcn.__name__))
    if answer==expected_answer:
        print(fcn.__name__, ' passed!  :-)\n')
    else:
        print(fcn.__name__, ' failed!  :-(\n')
        print('Expected ');print(expected_answer)
        print('But, received ');print(answer)


def test_solve_weighted_sokoban():
    print('<<  Testing {} >>\n'.format(fcn.__name__))
    answer = []
    expected_answer = []
    #Test 1
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_09.txt")
    expected_answer.append(396)
    t0 = time.time()
    result = solve_weighted_sokoban(wh)[1]
    answer.append(result)
    t1 = time.time()
    print('Test on warehouse 9')
    print ("It took: ",t1-t0, ' seconds')

    # Test 2
    wh.load_warehouse("./warehouses/warehouse_47.txt")
    expected_answer.append(179)
    t0 = time.time()
    result = solve_weighted_sokoban(wh)[1]
    answer.append(result)
    t1 = time.time()
    print('Test on warehouse 47')
    print ("It took: ",t1-t0, ' seconds')

    # Test 3
    wh.load_warehouse("./warehouses/warehouse_81.txt")
    expected_answer.append(376)
    t0 = time.time()
    result = solve_weighted_sokoban(wh)[1]
    answer.append(result)
    t1 = time.time()
    print('Test on warehouse 81')
    print ("It took: ",t1-t0, ' seconds')

#    # Test 4 To slow
#    wh.load_warehouse("./warehouses/warehouse_5n.txt")
#    expected_answer.append(None)
#    t0 = time.time()
#    result = solve_weighted_sokoban(wh)[1]
#    answer.append(result)
#    t1 = time.time()
#    print('Test on warehouse 9')
#    print ("It took: ",t1-t0, ' seconds')
#
    # Test 5 - Takes too long
#    wh.load_warehouse("./warehouses/warehouse_137.txt")
#    expected_answer.append(521)
#    t0 = time.time()
#    answer.append(solve_weighted_sokoban(wh))
#    t1 = time.time()
#    print('Test on warehouse 9')
#    print ("It took: ",t1-t0, ' seconds')


    fcn = solve_weighted_sokoban
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
    test_actions()
    test_results()
    test_check_elem_action_seq()
    test_h()
    test_solve_weighted_sokoban()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# - - - - - - - - - - - - - - - - -Code Cemetary- - - - - - - - - - - - - - - - 
