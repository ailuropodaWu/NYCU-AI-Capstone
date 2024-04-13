import sys
import random
import STcpClient

sys.path.append("../..")

from game_state import GameState


"""
選擇起始位置
選擇範圍僅限場地邊緣(至少一個方向為牆)

return: init_pos
init_pos=[x,y],代表起始位置
"""

DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


def InitPos(mapStat):
    init_pos = [0, 0]
    """
    Write your code here
    """
    bestLegalMove = 0
    for row in range(len(mapStat)):
        for col in range(len(mapStat[0])):
            if mapStat[row][col] != 0: 
                continue
            if (row == 0 or row == len(mapStat) - 1 or mapStat[row+1][col] == -1 or mapStat[row-1][col] == -1) or (col == 0 or col == len(mapStat[row]) - 1 or mapStat[row][col + 1] == -1 or mapStat[row][col - 1] == -1):
                legalMove = 0
                for dir in DIRECTION:
                    if dir == (0, 0): continue
                    if 0 <= row + dir[0] < len(mapStat) and 0 <= col + dir[1] < len(mapStat[0]) and mapStat[row + dir[0]][col + dir[1]] == 0:
                        if 0 in dir: legalMove += 0.5
                        legalMove += 1
                if legalMove > bestLegalMove:
                    bestLegalMove, init_pos = legalMove, [row, col]
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

    algorithm = "random"
    state = GameState(mapStat, sheepStat)

    if algorithm == "random":
        legalMoves = state.getLegalMoves(playerID)
        return random.choice(legalMoves) if legalMoves else None


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
