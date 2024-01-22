# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import Directions
import random, util

from game import Agent

from collections import deque


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    PACMAN_AGENT_INDEX = 0

    def __init__(self, evalFn='customEvaluation', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = customEvaluation
        self.depth = int(depth)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def maxValue(state, depth):
            maxScore = float("-inf")
            for nextAction in state.getLegalActions(self.PACMAN_AGENT_INDEX):
                maxScore = max(maxScore,
                               calculateScore(state.generateSuccessor(self.PACMAN_AGENT_INDEX, nextAction),
                                              depth, 1))
            return maxScore

        def minValue(state, depth, ghost_index):
            minScore = float("inf")
            for nextAction in state.getLegalActions(ghost_index):
                minScore = min(minScore,
                               calculateScore(state.generateSuccessor(ghost_index, nextAction),
                                              depth, ghost_index + 1))
            return minScore

        def calculateScore(state, depth, agent_index):
            agent_index %= gameState.getNumAgents()
            if agent_index == 0:
                depth += 1
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agent_index == self.PACMAN_AGENT_INDEX:
                return maxValue(state, depth)
            else:
                return minValue(state, depth, agent_index)

        action = Directions.STOP
        maxScore = float("-inf")
        for nextAction in gameState.getLegalActions():
            currentScore = calculateScore(gameState.generateSuccessor(self.PACMAN_AGENT_INDEX, nextAction), 0, 1)
            if currentScore >= maxScore:
                maxScore = currentScore
                action = nextAction

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def reverse_tuple(t):
    new_tuple = ()
    for i in range(len(t) - 1, -1, -1):
        new_tuple += (t[i],)
    return new_tuple


def find_food(GameMap, start):
    score = 0

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible movements: down, up, right, left
    num_rows, num_cols = len(GameMap), len(GameMap[0])

    h = [[float('-inf') for _ in range(num_cols)] for _ in range(num_rows)]

    visited = set()
    queue = deque([start])
    visited.add(start)

    fx, fy = start
    h[fx][fy] = 0

    while queue:
        current_cell = queue.popleft()
        x, y = current_cell
        if GameMap[x][y] == "o":
            return score + (10 / max(h[x][y], 1))
        elif GameMap[x][y] == ".":
            return score + (10 / max(h[x][y], 1))

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            # Check if the new coordinates are within the bounds of the grid
            if 0 <= new_x < num_rows and 0 <= new_y < num_cols:
                # Check if the new cell is not a blocked cell ("%") or goal cell ("G")
                if GameMap[new_x][new_y] not in ["%", "G"]:
                    new_cell = (new_x, new_y)
                    h[new_x][new_y] = h[x][y] + 1
                    if new_cell not in visited:
                        queue.append(new_cell)
                        visited.add(new_cell)
    return score


def find_scared_ghost(GameMap, start, ghostState):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible movements: down, up, right, left
    num_rows, num_cols = len(GameMap), len(GameMap[0])

    h = [[float('-inf') for _ in range(num_cols)] for _ in range(num_rows)]

    start = reverse_tuple(start)
    fx, fy = start
    fx = int(fx)
    fy = int(fy)
    start = (fx, fy)

    visited = set()
    queue = deque([start])
    visited.add(start)

    fx, fy = start
    h[fx][fy] = 0

    while queue:
        current_cell = queue.popleft()
        x, y = current_cell

        if GameMap[x][y] in ["<", ">", "v", "^"]:
            if h[x][y] <= ghostState.scaredTimer * 1.7:
                return h[x][y]
            else:
                return False

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            # Check if the new coordinates are within the bounds of the grid
            if 0 <= new_x < num_rows and 0 <= new_y < num_cols:
                # Check if the new cell is not a blocked cell ("%") or goal cell ("G")
                if GameMap[new_x][new_y] not in ["%", "G"]:
                    new_cell = (new_x, new_y)
                    h[new_x][new_y] = h[x][y] + 1
                    if new_cell not in visited:
                        queue.append(new_cell)
                        visited.add(new_cell)
    return False


def customEvaluation(currentGameState):
    if currentGameState.isWin() or currentGameState.isLose():
        return currentGameState.getScore()
    pacmanPosition = currentGameState.getPacmanPosition()  # 0->Column 1->
    GameMap = currentGameState.__str__().splitlines()[:-1][::-1]

    num_rows, num_cols = len(GameMap), len(GameMap[0])

    # checks if it can eat a ghost
    for i in range(currentGameState.getNumAgents() - 1):
        ghostState = currentGameState.data.agentStates[i + 1]
        ghostPosition = ghostState.configuration.getPosition()
        if ghostState.scaredTimer > 0:
            result = find_scared_ghost(GameMap, ghostPosition, ghostState)
            if result:
                return currentGameState.getScore() + (200 / max(result, 1))

    # calculate remaining food
    restFood = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if GameMap[i][j] in ['.', 'o']:
                restFood += 1

    pacmanPosition = reverse_tuple(pacmanPosition)
    scoreEvl = find_food(GameMap, pacmanPosition)

    return scoreEvl + currentGameState.getScore() - restFood * 2


# Abbreviation
better = betterEvaluationFunction
