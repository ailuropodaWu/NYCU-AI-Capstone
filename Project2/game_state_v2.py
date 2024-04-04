import numpy as np

from game_state import GameState as BaseGameState


DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


class GameState(BaseGameState):
    def __init__(self, _mapStat, _sheepStat, _playerNum=4):
        super().__init__(_mapStat, _sheepStat, _playerNum)

    def evaluate(self, id):
        self._calculateScore()
        rank = self.getRank(id)
        fourNeighbors, eightNeighbors = 0, 0
        for move in self.getLegalMoves(id):
            if move[-1] % 2 == 0: fourNeighbors += 1
            else: eightNeighbors += 1
        sheeps = self.sheep[self.mapStat == id]
        score_part = self.scores[id - 1] / 32 # 16 ^ 1.25 = 32
        neighbors_part = (0.7 * fourNeighbors + 0.3 * eightNeighbors) / 4 # prefer 4 neighbors over 8 neighbors
        rank_part = 0.3 * (1 - rank / 4) # prefer higher rank
        sheeps_part = -(0.5 * np.mean(sheeps) + 0.5 * np.var(sheeps)) / 16 # avoid sheep to be too concentrated
        return np.mean([score_part, neighbors_part, rank_part, sheeps_part])
