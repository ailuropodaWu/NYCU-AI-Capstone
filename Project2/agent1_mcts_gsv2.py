import sys
import random
import numpy as np
import STcpClient

from time import time

sys.path.append("../..")

from game_state import weightedMap
from game_state_v2 import GameState
from mcts import MCTS, Node


"""
選擇起始位置
選擇範圍僅限場地邊緣(至少一個方向為牆)

return: init_pos
init_pos=[x,y],代表起始位置
"""

DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


class SheepGame(Node):
    def __init__(self, state: GameState, playerID: int, move=None):
        super().__init__()
        self.state = state
        self.player_num = state.playerNum
        self.player_id = playerID
        self.children = []
        self.move = move

    def find_children(self):
        return [SheepGame(self.state.getNextState(move, self.player_id), self.player_id, move) for move in self.state.getLegalMoves(self.player_id)]

    def find_random_child(self):
        random_move = random.choice(self.state.getLegalMoves(self.player_id))
        return SheepGame(self.state.getNextState(random_move, self.player_id), self.player_id, random_move)

    def is_terminal(self):
        return self.state.noMove(self.player_id)

    def reward(self):
        return self.state.evaluate(self.player_id)

    def __hash__(self):
        return hash((self.state, self.player_id))

    def __eq__(self, other):
        return hash(self) == hash(other)


def InitPos(mapStat):
    init_pos = [0, 0]
    """
    Write your code here
    """
    mapStat = np.array(mapStat)
    available = mapStat == 0
    neighbors = weightedMap(mapStat, kernel=(3, 3), weights=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), filling=0)
    available[neighbors == 0] = False
    weighted_map = weightedMap(mapStat)
    weighted_map[~available] = np.inf
    init_pos = np.unravel_index(np.argmin(weighted_map), mapStat.shape)
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

    max_iter = int(1e5)
    mcts = MCTS()
    game_state = GameState(mapStat, sheepStat)
    root = SheepGame(game_state, playerID)

    if game_state.noMove(playerID): return [(0, 0), 0, 1]
    start = time()
    for i in range(max_iter):
        mcts.do_rollout(root)
        if time() - start > 2.8: break

    best_state = mcts.choose(root)

    print(f"Iterations: {i} times")
    print(f"Best reward for player {playerID}: {best_state.state.evaluate(playerID):.4f}")
    print(f"Current score: {game_state.scores[playerID - 1]}")

    return best_state.move


if __name__ == "__main__":
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
