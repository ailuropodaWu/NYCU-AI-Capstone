import numpy as np

from game_state import GameState as BaseGameState, findConnected, weightedMap


DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


def get_bbox(array):
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def villain_map(id, mapStat, kernel=(5, 5), weights=None, filling=-1):
    enemy = (2, 4) if id == 1 or id == 3 else (1, 3)
    kw, kh = kernel
    weights = np.ones(kernel) if weights is None else weights
    assert weights.shape == kernel
    mapStat = np.array(mapStat)

    mapStat = np.where((mapStat == enemy[0]) | (mapStat == enemy[1]), 1, 0) #  Set cells to 1 where the value is either 2 or 4, otherwise set to 0

    mapStat = np.pad(mapStat, ((kw // 2, kw // 2), (kh // 2, kh // 2)), "constant", constant_values=filling)
    mapStat = np.abs(mapStat)
    
    sub_matrices = np.lib.stride_tricks.sliding_window_view(mapStat, kernel)
    return np.einsum("ij,klij->kl", weights, sub_matrices)

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
        mapStat = self.mapStat.copy()
        mapStat[mapStat > 0] = -1
        mapStat[self.mapStat == id] = self.sheep[self.mapStat == id]
        space_part = -weightedMap(mapStat, kernel=(3, 3), weights=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9)[self.mapStat == id].mean() / 24 # avoid too many obstacles around, 24 = 16 + 8
        
        threat_part = villain_map(id, mapStat, kernel=(3, 3), weights=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9)[self.mapStat == id].mean() / 24 # avoid too many obstacles around, 24 = 16 + 8

        rewards = np.asarray([score_part, rank_part, sheeps_part, space_part, threat_part])
        rewards /= np.linalg.norm(rewards)
        return np.dot(rewards, np.ones((5,)) / 5)

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
                    legalMoves.extend([[(row, col), sheep_num, dir_i + 1] for sheep_num in range(1, int(self.sheep[row, col]))])
        return legalMoves

class EndGameState(GameState):
    def __init__(self, _mapStat, _sheepStat, _maxSheep, _playerNum=4):
        super().__init__(_mapStat, _sheepStat, _maxSheep, _playerNum)
    def evaluate(self, id):
        self._calculateScore()
        rank = self.getRank(id)
        # boundingBox = get_bbox(self.mapStat == id)
        # area_part = -(0.5 * (boundingBox[1] - boundingBox[0] + 1) * (boundingBox[3] - boundingBox[2] + 1) / 144) # avoid area too large
        connected_part = np.max([len(r) for r in findConnected(self, id)]) / self.maxSheep # prefer large connected regions
        score_part = self.scores[id - 1] / (self.maxSheep ** 1.25) # 16 ^ 1.25 = 32
        rank_part = 1 - rank / 4 # prefer higher rank
        return np.dot([connected_part, score_part, rank_part], [0.4, 0.4, 0.2])
        