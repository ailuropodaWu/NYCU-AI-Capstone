import copy
import numpy as np
from game_state import GameState as BaseGameState


DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


class GameState(BaseGameState):
    def __init__(self, _mapStat, _sheepStat, _playerNum=4):
        self.mapStat = np.asarray(_mapStat)
        self.sheep = np.asarray(_sheepStat)
        self.scores = np.zeros((_playerNum,))
        self.playerNum = _playerNum
        self._getAgentState()
        

    def _getAgentState(self):
        '''
            self.agentStat:
                [
                    [[(row, col), sheepNum], ...],  \
                    [[(row, col), sheepNum], ...],  playerNum  
                                 ...                /
                ]
            self.legalMove:
                {
                    (row, col): [dir, ...],
                    ...
                }
        
        '''
        agentStat = []
        self.legalMove = {}
        for _ in range(self.playerNum):
            agentStat.append([])
        for row, col in np.ndindex(self.mapStat.shape):
            if self.mapStat[row, col] <= 0 or self.sheep[row, col] <= 1:
                continue
            id = int(self.mapStat[row, col])

            
            for dir_i in range(len(DIRECTION)):
                if dir_i == 4: continue
                dir = DIRECTION[dir_i]
                if 0 <= row + dir[0] < len(self.mapStat) and \
                    0 <= col + dir[1] < len(self.mapStat[0]) and \
                    self.mapStat[row + dir[0], col + dir[1]] == 0:
                    if [(row, col), self.sheep[row, col]] not in agentStat[id - 1]:
                        agentStat[id - 1].append([(row, col), self.sheep[row, col]])
                    if (row, col) not in self.legalMove.keys():
                        self.legalMove[(row, col)] = []
                        
                    self.legalMove[(row, col)].append(dir_i + 1)
        self.agentStat = agentStat

    def evaluate(self, id):
        self._calculateScore()
        ranks = np.argsort(-self.scores)
        legalMoves = self.getLegalMoves(id)
        goodMoveNum = 0
        for move in legalMoves:
            if move[-1] % 2 == 0:
                goodMoveNum += 1
        score = self.scores[id - 1]
        rank = np.where(ranks == id - 1)[0][0] + 1
        eval = 0.4 * score + 1 / rank + 0.1 * goodMoveNum 
        return eval

    def noMove(self, id):
        return len(self.agentStat[id - 1]) == 0

    def gameOver(self):
        for id in range(1, self.playerNum + 1):
            if not self.noMove(id):
                return False
        return True

    def getLegalMoves(self, id):
        legalMoves = []

        for state in self.agentStat[id - 1]:
            if state[1] <= 1:
                continue
            pos = state[0]
            row, col = pos
            split = int(self.sheep[row, col] // 2)
            if pos in self.legalMove.keys():
                for dir_i in self.legalMove[pos]:
                    legalMoves.append([pos, split, dir_i])
            else:
                raise("Key error")
        return legalMoves

    def getNextState(self, move, id):
        newState = copy.deepcopy(self)
        pos, split, dir_i = move
        row, col = pos
        if self.mapStat[row, col] != id or self.sheep[row, col] < split:
            raise("State error")
        dir = DIRECTION[dir_i - 1]
        newState.sheep[row, col] -= split
        remove = None
        for i in range(len(newState.agentStat[id - 1])):
            state = newState.agentStat[id - 1][i]
            if state[0] == (row, col):
                newState.agentStat[id - 1][i][1] -= split
                if newState.agentStat[id - 1][i][1] <= 1:
                    remove = newState.agentStat[id - 1][i]
                break
        if remove: 
            newState.agentStat[id - 1].remove(remove)

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
        for dir_i in range(len(DIRECTION)):
            if dir_i == 4: continue
            dir = DIRECTION[dir_i]
            if 0 <= row + dir[0] < len(newState.mapStat) and \
                0 <= col + dir[1] < len(newState.mapStat[0]):
                if split > 1 and newState.mapStat[row + dir[0], col + dir[1]] == 0:
                    if [(row, col), split] not in newState.agentStat[id - 1]:
                        newState.agentStat[id - 1].append([(row,col), split])
                    if (row, col) not in newState.legalMove.keys():
                        newState.legalMove[(row, col)] = []
                    newState.legalMove[(row, col)].append(dir_i + 1)
                elif newState.mapStat[row + dir[0], col + dir[1]] > 0 and newState.sheep[row + dir[0], col + dir[1]] > 1:
                    neighbor_id = int(newState.mapStat[row + dir[0], col + dir[1]])
                    r, c = row + dir[0], col + dir[1]
                    if (r, c) not in newState.legalMove.keys():
                        raise("Legal move update error")
                    newState.legalMove[(r, c)].remove(9 - dir_i)
                    if len(newState.legalMove[(r, c)]) == 0:
                        for state in newState.agentStat[neighbor_id - 1]:
                            if state[0] == (r, c):
                                newState.agentStat[neighbor_id - 1].remove(state)
                                break
        return newState
