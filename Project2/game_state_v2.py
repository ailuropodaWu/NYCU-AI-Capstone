import numpy as np

from game_state import GameState as BaseGameState, weightedMap


DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


def get_bbox(array):
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


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
