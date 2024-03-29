import STcpClient
import numpy as np
import random
import copy

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
direction = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


def InitPos(mapStat):
    init_pos = [0, 0]
    '''
        Write your code here

    '''
    bestLegalMove = 0
    for row in range(len(mapStat)):
        for col in range(len(mapStat[0])):
            if mapStat[row][col] != 0: 
                continue
            if (row == 0 or row == len(mapStat) - 1 or mapStat[row+1][col] == -1 or mapStat[row-1][col] == -1) or \
                (col == 0 or col == len(mapStat[row]) - 1 or mapStat[row][col+1] == -1 or mapStat[row][col-1] == -1):
                legalMove = 0
                for dir in direction:
                    if dir == (0, 0):
                        continue
                    if 0 <= row + dir[0] < len(mapStat) and \
                        0 <= col + dir[1] < len(mapStat[0]) and \
                        mapStat[row+dir[0]][col+dir[1]] == 0:
                        if 0 in dir:
                            legalMove += 0.5
                        legalMove += 1
                if legalMove > bestLegalMove:
                    bestLegalMove, init_pos = legalMove, [row, col]
    return init_pos


'''
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
'''
def GetStep(playerID, mapStat, sheepStat):
    step = [(0, 0), 0, 1]
    '''
    Write your code here
    
    '''

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
            legal_moves = self.get_legal_moves(id)
            good_move_num = 0
            for move in legal_moves:
                if move[-1] % 2 == 0:
                    good_move_num += 1
            score = self.scores[id]
            rank = np.where(ranks == id)[0][0] + 1
            eval = 0.4 * score + 1 / rank + 0.1 * good_move_num 
            return eval
        
        def gameOver(self, id):
            for row in range(len(self.mapStat)):
                for col in range(len(self.mapStat[0])):
                    for dir_i in range(len(direction)):
                        if dir_i == 4:
                            continue
                        dir = direction[dir_i]
                        if self.mapStat[row][col] == id and self.sheep[row][col] > 1 and \
                            0 <= row + dir[0] < len(self.mapStat) and \
                            0 <= col + dir[1] < len(self.mapStat[0]) and \
                            self.mapStat[row+dir[0]][col+dir[1]] == 0:
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

        def get_legal_moves(self, id):
            legal_moves = []
            for row in range(len(self.mapStat)):
                for col in range(len(self.mapStat[0])):
                    if self.mapStat[row][col] != id or self.sheep[row][col] <= 1:  # Select cells with more than one sheep
                        continue
                    for dir_i in range(len(direction)):
                        if dir_i == 4:
                            continue
                        dir = direction[dir_i]
                        if 0 <= row + dir[0] < len(self.mapStat) and \
                            0 <= col + dir[1] < len(self.mapStat[0]) and \
                            self.mapStat[row+dir[0]][col+dir[1]] == 0:
                            # only consider half split
                            legal_moves.append([(row, col), int(self.sheep[row][col] // 2), dir_i+1])
            return legal_moves


        def apply_move(self, move, id):
            new_state = copy.deepcopy(self)
            pos, split, dir_i = move
            row, col = pos
            if self.mapStat[row][col] != id or self.sheep[row][col] < split:
                raise("State error")
            dir = direction[dir_i-1]
            new_state.sheep[row][col] -= split
            end = False
            while not end:
                if 0 <= row + dir[0] < len(self.mapStat) and \
                    0 <= col + dir[1] < len(self.mapStat[0]) and \
                    new_state.mapStat[row+dir[0]][col+dir[1]] == 0:
                    row += dir[0]
                    col += dir[1]
                else:
                    end = True
            if new_state.sheep[row][col] != 0 or new_state.mapStat[row][col] != 0:
                raise("Move error")
            new_state.sheep[row][col] = split
            new_state.mapStat[row][col] = id

            return new_state

    

    maxDepth = 1
    def minimax(game_state: GameState, depth, id, alpha, beta):
        if depth == 0 or game_state.gameOver(id):
            return game_state.evaluate(id)
        legal_moves = game_state.get_legal_moves(id)
        evalNodes = []
        if id == playerID:
            ret = float("-inf")
            next_id = id + 1
            if id == game_state.playerNum:
                next_id = 1
            for move in legal_moves:
                new_state = game_state.apply_move(move, id)
                eval = minimax(new_state, depth, next_id, alpha, beta)
                evalNodes.append(eval)
                alpha = max(alpha, max(evalNodes))
                ret = max(evalNodes)
                if ret >= beta:
                    break
            if depth == maxDepth:
                for i in range(len(evalNodes)): 
                    if evalNodes[i] == max(evalNodes):
                        return legal_moves[i]
            else:
                return ret
        else:
            next_id = id + 1
            if id == game_state.playerNum:
                next_id = 1
            if next_id == playerID:
                depth -= 1
            for move in legal_moves:
                new_state = game_state.apply_move(move, id)
                eval = minimax(new_state, depth, next_id, alpha, beta)
                evalNodes.append(eval)
                beta = min(beta, min(evalNodes))
                if min(evalNodes) <= alpha:
                    return min(evalNodes)
            return min(evalNodes)
        
    state = GameState(mapStat, sheepStat)

    
    return minimax(state, maxDepth, 1, float("-inf"), float("inf"))


    # my_stat = []
    # best_distance = 0
    # print(mapStat)
    # for r in range(len(mapStat)):
    #     for c in range(len(mapStat[r])):
    #         if mapStat[r][c] == playerID:
    #             my_stat.append(((r, c), sheepStat[r][c]))
    # for stat in my_stat:
    #     if stat[1] <= 1:
    #         continue
    #     for dir_i in range(len(direction)):
    #         if dir_i == 4:
    #             continue
    #         dir = direction[dir_i]
    #         distance = 0
    #         end = False
    #         r, c = stat[0][0], stat[0][1]
    #         while not end:
    #             if (r + dir[0]) >= 0 and (r + dir[0]) <= len(mapStat)-1 and (c + dir[1]) >= 0 and (c + dir[1]) <= len(mapStat[r])-1 \
    #                   and mapStat[r + dir[0]][c + dir[1]] == 0:
    #                 distance += 1
    #                 r += dir[0]
    #                 c += dir[1]
    #             else:
    #                 end = True
    #         if distance > best_distance:
    #             best_distance = distance
    #             step = [stat[0], int(stat[1] // 2), dir_i+1]
    #             print(step)
    # return step


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
