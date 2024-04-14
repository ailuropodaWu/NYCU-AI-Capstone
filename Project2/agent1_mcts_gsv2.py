"""
Team ID: 25
Team name: Whatever
Team members: 110550036 張家維, 110550014 吳權祐, 110550100 廖奕瑋
"""


import sys
import random
import numpy as np
import copy
import STcpClient
from abc import ABC, abstractmethod
from collections import defaultdict
import math

from time import time




"""
選擇起始位置
選擇範圍僅限場地邊緣(至少一個方向為牆)

return: init_pos
init_pos=[x,y],代表起始位置
"""

DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        self._backpropagate(path)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _backpropagate(self, path):
        "Send the reward back up to the ancestors of the leaf"
        ego_reward = path[-1].state.evaluate(path[-1].ego_player)
        reward = path[-1].reward()
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward if node.player_id == node.ego_player else ego_reward - reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True

class SheepGame(Node):
    def __init__(self, state, playerID: int, egoPlayer: int, move=None):
        super().__init__()
        self.state = state
        self.player_num = state.playerNum
        self.player_id = playerID
        self.ego_player = egoPlayer
        self.children = []
        self.move = move

    def find_children(self):
        next_player_id = self.player_id % self.player_num + 1
        children = [SheepGame(self.state.getNextState(move, self.player_id), next_player_id, self.ego_player, move) for move in self.state.getLegalMoves(self.player_id)]
        random.shuffle(children)
        return children

    def find_random_child(self):
        next_player_id = self.player_id % self.player_num + 1
        random_move = random.choice(self.state.getLegalMoves(self.player_id))
        return SheepGame(self.state.getNextState(random_move, self.player_id), next_player_id, self.ego_player, random_move)

    def is_terminal(self):
        return self.state.gameOver()

    def reward(self):
        return self.state.evaluate(self.player_id)

    def __hash__(self):
        return hash((self.state, self.player_id))

    def __eq__(self, other):
        return hash(self) == hash(other)



class BaseGameState:
    def __init__(self, _mapStat, _sheepStat, _maxSheep=32, _playerNum=4):
        self.mapStat = np.asarray(_mapStat)
        self.sheep = np.asarray(_sheepStat)
        self.scores = np.zeros((_playerNum,))
        self.playerNum = _playerNum
        self.maxSheep = _maxSheep

    def evaluate(self, id):
        id -= 1
        self._calculateScore()
        ranks = np.argsort(-self.scores)
        legalMoves = self.getLegalMoves(id)
        goodMoveNum = 0
        for move in legalMoves:
            if move[-1] % 2 == 0:
                goodMoveNum += 1
        score = self.scores[id]
        rank = np.where(ranks == id)[0][0] + 1
        eval = 0.4 * score + 1 / rank + 0.1 * goodMoveNum 
        return eval

    def getScore(self, id):
        return self.scores[id - 1]

    def getRank(self, id):
        ranks = np.argsort(-self.scores)
        return np.where(ranks == id - 1)[0][0] + 1

    def noMove(self, id):
        for row, col in np.ndindex(self.mapStat.shape):
            for dir in DIRECTION:
                if dir[0] == 0 and dir[1] == 0: continue
                if self.mapStat[row, col] == id and self.sheep[row, col] > 1 and \
                    0 <= row + dir[0] < len(self.mapStat) and \
                    0 <= col + dir[1] < len(self.mapStat[0]) and \
                    self.mapStat[row + dir[0], col + dir[1]] == 0:
                    return False
        return True

    def gameOver(self):
        for id in range(1, self.playerNum + 1):
            if not self.noMove(id):
                return False
        return True

    def _calculateScore(self):
        for i in range(self.playerNum):
            id = i + 1
            connectedRegions = findConnected(self, id)
            self.scores[i] = sum(len(region) ** 1.25 for region in connectedRegions)
            
    def getLegalMoves(self, id):
        legalMoves = []
        for row, col in np.ndindex(self.mapStat.shape):
            # Select cells with more than one sheep
            if self.mapStat[row, col] != id or self.sheep[row, col] <= 1: continue
            for dir_i, dir in enumerate(DIRECTION):
                if dir_i == 4: continue
                if 0 <= row + dir[0] < len(self.mapStat) and \
                    0 <= col + dir[1] < len(self.mapStat[0]) and \
                    self.mapStat[row+dir[0], col+dir[1]] == 0:
                    # only consider half split
                    legalMoves.append([(row, col), int(self.sheep[row, col] // 2), dir_i + 1])
        return legalMoves

    def getNextState(self, move, id):
        newState = copy.deepcopy(self)
        pos, split, dir_i = move
        row, col = pos
        if self.mapStat[row, col] != id or self.sheep[row, col] < split:
            raise("State error")
        dir = DIRECTION[dir_i - 1]
        newState.sheep[row, col] -= split
        end = False
        while not end:
            if 0 <= row + dir[0] < len(self.mapStat) and \
                0 <= col + dir[1] < len(self.mapStat[0]) and \
                newState.mapStat[row + dir[0], col + dir[1]] == 0:
                row += dir[0]
                col += dir[1]
            else:
                end = True
        if newState.sheep[row, col] != 0 or newState.mapStat[row, col] != 0:
            raise("Move error")
        newState.sheep[row, col] = split
        newState.mapStat[row, col] = id

        return newState


class GameState(BaseGameState):
    def __init__(self, _mapStat, _sheepStat, _maxSheep, _playerNum=4):
        super().__init__(_mapStat, _sheepStat, _maxSheep, _playerNum)

    def evaluate(self, id):
        self._calculateScore()
        rank = self.getRank(id)
        sheeps = self.sheep[self.mapStat == id]
        score_part = self.scores[id - 1] / (self.maxSheep ** 1.25) # 16 ^ 1.25 = 32
        rank_part = 1 - rank / 4 # prefer higher rank
        sheeps_part = (-np.max(sheeps) - 1) / (self.maxSheep - 1) # avoid sheep to be too concentrated
        moves_part = len(self.getLegalMoves(id)) / (np.count_nonzero(self.sheep[self.mapStat == id] - 1) * 8 + 1) # prefer more legal moves
        rewards = np.asarray([score_part, rank_part, sheeps_part, moves_part])
        rewards /= np.linalg.norm(rewards)
        return np.dot(rewards, np.ones(4) / 4)

    def getLegalMoves(self, id):
        legalMoves = []
        for row, col in np.ndindex(self.mapStat.shape):
            # Select cells with more than one sheep
            if self.mapStat[row, col] != id or self.sheep[row, col] <= 1: continue
            for dir_i, dir in enumerate(DIRECTION):
                if dir_i == 4: continue
                if 0 <= row + dir[0] < len(self.mapStat) and \
                    0 <= col + dir[1] < len(self.mapStat[0]) and \
                    self.mapStat[row+dir[0], col+dir[1]] == 0:
                    possible_moves = set([1, int(self.sheep[row, col] // 2), int(self.sheep[row, col]) - 1])
                    legalMoves.extend([((row, col), m, dir_i + 1) for m in possible_moves])
        return legalMoves

def weightedMap(mapStat, kernel=(5, 5), weights=None, filling=-1):
    kw, kh = kernel
    weights = np.ones(kernel) if weights is None else weights
    assert weights.shape == kernel
    mapStat = np.array(mapStat)
    mapStat = np.pad(mapStat, ((kw // 2, kw // 2), (kh // 2, kh // 2)), "constant", constant_values=filling)
    mapStat = np.abs(mapStat)
    sub_matrices = np.lib.stride_tricks.sliding_window_view(mapStat, kernel)
    return np.einsum("ij,klij->kl", weights, sub_matrices)


def findConnected(gameState: GameState, id):
    visited = set()
    connectedRegions = []

    def dfs(row, col, region):
        if row < 0 or row >= len(gameState.mapStat) or \
            col < 0 or col >= len(gameState.mapStat[0]) or \
            (row, col) in visited:
            return
        if gameState.mapStat[row, col] == id:
            visited.add((row, col))
            region.append((row, col))
            dfs(row + 1, col, region)
            dfs(row - 1, col, region)
            dfs(row, col + 1, region)
            dfs(row, col - 1, region)

    for row, col in np.ndindex(gameState.mapStat.shape):
        if gameState.mapStat[row, col] == id and (row, col) not in visited:
            region = []
            dfs(row, col, region)
            connectedRegions.append(region)
    return connectedRegions

def InitPos(mapStat):
    init_pos = [0, 0]
    """
    Write your code here
    """
    mapStat = np.array(mapStat)
    available = mapStat == 0
    neighbors = weightedMap(mapStat, kernel=(3, 3), weights=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), filling=0)
    available[neighbors == 0] = False
    weighted_map = weightedMap(mapStat)
    weighted_map[~available] = np.inf
    init_pos = np.unravel_index(np.argmin(weighted_map), mapStat.shape)
    print(weighted_map)
    return init_pos


"""
產出指令

input: 
playerID: 你在此局遊戲中的角色(1~4)
mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
            0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

return Step
Step : 3 elements, [(x,y), m, dir]
        x, y 表示要進行動作的座標 
        m = 要切割成第二群的羊群數量
        dir = 移動方向(1~9),對應方向如下圖所示
        1 2 3
        4 X 6
        7 8 9
"""

def GetStep(playerID, mapStat, sheepStat):
    """
    Write your code here
    """

    mcts = MCTS()
    maxSheep = 16
    game_state = GameState(mapStat, sheepStat, maxSheep)
    root = SheepGame(game_state, playerID, playerID)
    max_iter = int(1e5)

    if game_state.noMove(playerID): return [(0, 0), 0, 1]
    start = time()
    for i in range(max_iter):
        mcts.do_rollout(root)
        if time() - start > 2.8: break

    best_state = mcts.choose(root)

    print(f"Iterations: {i} times")
    print(f"Current score: {game_state.scores[playerID - 1]:.4f}")

    return best_state.move


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
