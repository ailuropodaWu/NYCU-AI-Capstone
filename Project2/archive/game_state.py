import numpy as np
import random 
import copy

DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))

class GameState:
    def __init__(self, _mapStat, _sheepStat):
        self.mapStat = _mapStat
        self.sheep = _sheepStat
        self.playerNum = 4
        self.scores = np.zeros((self.playerNum,))
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
        return self.scores[id-1]
    
    def getRank(self, id):
        ranks = np.argsort(-self.scores)
        return np.where(ranks == id - 1)[0][0] + 1
    
    def noMove(self, id):
        for row in range(len(self.mapStat)):
            for col in range(len(self.mapStat[0])):
                for dir_i in range(len(DIRECTION)):
                    if dir_i == 4:
                        continue
                    dir = DIRECTION[dir_i]
                    if self.mapStat[row][col] == id and self.sheep[row][col] > 1 and \
                        0 <= row + dir[0] < len(self.mapStat) and \
                        0 <= col + dir[1] < len(self.mapStat[0]) and \
                        self.mapStat[row + dir[0]][col + dir[1]] == 0:
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
            connectedRegions = self._findConnected(id)
            self.scores[i] = round(sum(len(region) ** 1.25 for region in connectedRegions))
    def _findConnected(self, id):
        visited = set()
        regions = []

        def dfs(row, col, region):
            if row < 0 or row >= len(self.mapStat) or \
                col < 0 or col >= len(self.mapStat[0]) or \
                (row, col) in visited:
                return
            if self.mapStat[row][col] == id:
                visited.add((row, col))
                region.append((row, col))
                dfs(row + 1, col, region)
                dfs(row - 1, col, region)
                dfs(row, col + 1, region)
                dfs(row, col - 1, region)

        for row in range(len(self.mapStat)):
            for col in range(len(self.mapStat[0])):
                if self.mapStat[row][col] == id and (row, col) not in visited:
                    region = []
                    dfs(row, col, region)
                    regions.append(region)
        return regions

    def getLegalMoves(self, id):
        legalMoves = []
        for row in range(len(self.mapStat)):
            for col in range(len(self.mapStat[0])):
                if self.mapStat[row][col] != id or self.sheep[row][col] <= 1:  # Select cells with more than one sheep
                    continue
                for dir_i in range(len(DIRECTION)):
                    if dir_i == 4:
                        continue
                    dir = DIRECTION[dir_i]
                    if 0 <= row + dir[0] < len(self.mapStat) and \
                        0 <= col + dir[1] < len(self.mapStat[0]) and \
                        self.mapStat[row+dir[0]][col+dir[1]] == 0:
                        # only consider half split
                        legalMoves.append([(row, col), int(self.sheep[row][col] // 2), dir_i+1])
        return legalMoves


    def getNextState(self, move, id):
        newState = copy.deepcopy(self)
        pos, split, dir_i = move
        row, col = pos
        if self.mapStat[row][col] != id or self.sheep[row][col] < split:
            raise("State error")
        dir = DIRECTION[dir_i - 1]
        newState.sheep[row][col] -= split
        end = False
        while not end:
            if 0 <= row + dir[0] < len(self.mapStat) and \
                0 <= col + dir[1] < len(self.mapStat[0]) and \
                newState.mapStat[row + dir[0]][col + dir[1]] == 0:
                row += dir[0]
                col += dir[1]
            else:
                end = True
        if newState.sheep[row][col] != 0 or newState.mapStat[row][col] != 0:
            raise("Move error")
        newState.sheep[row][col] = split
        newState.mapStat[row][col] = id

        return newState
