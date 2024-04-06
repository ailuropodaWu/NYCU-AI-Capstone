import copy
import numpy as np


DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


class GameState:
    def __init__(self, _mapStat, _sheepStat, _playerNum=4):
        self.mapStat = np.asarray(_mapStat)
        self.sheep = np.asarray(_sheepStat)
        self.scores = np.zeros((_playerNum,))
        self.playerNum = _playerNum

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
            self.scores[i] = np.round(np.sum(len(region) ** 1.25 for region in connectedRegions))
            
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


def weightedMap(mapStat, kernel=(5, 5), weights=None):
    kw, kh = kernel
    weights = np.ones(kernel) if weights is None else weights
    assert weights.shape == kernel
    mapStat = np.array(mapStat).T
    mapStat = np.pad(mapStat, ((kw // 2, kw // 2), (kh // 2, kh // 2)), "constant", constant_values=-1)
    mapStat = np.abs(mapStat)
    sub_matrices = np.lib.stride_tricks.sliding_window_view(mapStat, kernel)
    return np.einsum("ij,klij->kl", weights, sub_matrices).T


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
