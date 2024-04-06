import STcpClient
import numpy as np
import random
import copy

import sys
sys.path.append('../..')
from game_state import GameState

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
DIRECTION = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


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
                (col == 0 or col == len(mapStat[row]) - 1 or mapStat[row][col + 1] == -1 or mapStat[row][col - 1] == -1):
                legalMove = 0
                for dir in DIRECTION:
                    if dir == (0, 0):
                        continue
                    if 0 <= row + dir[0] < len(mapStat) and \
                        0 <= col + dir[1] < len(mapStat[0]) and \
                        mapStat[row + dir[0]][col + dir[1]] == 0:
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

    
    

    maxDepth = 2
    def minimax(gameState: GameState, depth, id, alpha, beta):
        if depth == 0 or gameState.noMove(id):
            return gameState.evaluate(id)
        legalMoves = gameState.getLegalMoves(id)
        evalNodes = []
        if id == playerID:
            ret = float("-inf")
            nextID = id + 1
            if id == gameState.playerNum:
                nextID = 1
            for move in legalMoves:
                newState = gameState.getNextState(move, id)
                eval = minimax(newState, depth, nextID, alpha, beta)
                evalNodes.append(eval)
                alpha = max(alpha, max(evalNodes))
                ret = max(evalNodes)
                if ret >= beta:
                    break
            if depth == maxDepth:
                for i in range(len(evalNodes)): 
                    if evalNodes[i] == max(evalNodes):
                        return legalMoves[i]
            else:
                return ret
        else:
            nextID = id + 1
            if id == gameState.playerNum:
                nextID = 1
            if nextID == playerID:
                depth -= 1
            for move in legalMoves:
                newState = gameState.getNextState(move, id)
                eval = minimax(newState, depth, nextID, alpha, beta)
                evalNodes.append(eval)
                beta = min(beta, min(evalNodes))
                if min(evalNodes) <= alpha:
                    return min(evalNodes)
            return min(evalNodes)
    
    class Node:
        def __init__(self, state: GameState, parent=None, move=None):
            self.state = state
            self.parent = parent
            self.move = move
            self.children = []
            self.visits = 0
            self.rank = 0

        def isLeaf(self):
            return len(self.children) == 0

        def selectChild(self):
            # Implement UCB1 selection strategy
            if self.isLeaf():
                self.expand()
                return random.choice(self.children)
            else:
                ucbValues = []
                for child in self.children:
                    if child.visits:
                        value = (child.rank / child.visits) + np.sqrt(2 * np.log(self.visits) / child.visits)
                    else:
                        value = float('inf')
                ucbValues.append(value)
                return self.children[np.argmax(ucbValues)]
            

        def expand(self):
            # Expand the node by adding child nodes for each legal move
            legalMoves = self.state.getLegalMoves(playerID)
            for move in legalMoves:
                newState = self.state.getNextState(move, playerID)
                id = playerID
                for _ in range(1, self.state.playerNum):
                    id = id % self.state.playerNum + 1
                    otherLegalMoves = newState.getLegalMoves(id)
                    if not otherLegalMoves:
                        continue
                    otherMove = random.choice(otherLegalMoves)
                    newState = newState.getNextState(otherMove, id)
                childNode = Node(newState, self, move)
                self.children.append(childNode)

        def backpropagate(self, rank):
            # Update visit and win counts of nodes in the path from root to this node
            self.visits += 1
            self.rank += 1/ rank
            if self.parent is not None:
                self.parent.backpropagate(rank)

        def bestChild(self):
            # Return the best child node based on UCB1 formula
            if not self.children:
                raise ValueError("No children to select from.")
            return max(self.children, key=lambda c: c.visits)
        

    def simulate(gameState: GameState):
        passID = []
        id = playerID
        while not gameState.gameOver():
            if id in passID:
                id = (id % gameState.playerNum) + 1
                continue
            legalMoves = gameState.getLegalMoves(id)
            if not legalMoves:
                passID.append(id)
                id = (id % gameState.playerNum) + 1
                continue
            move = random.choice(legalMoves)
            gameState = gameState.getNextState(move, id)
            id = (id % gameState.playerNum) + 1
        rank = gameState.getRank(playerID) if gameState.gameOver() else None
        return rank

    def MCTS(gameState: GameState, maxSimulation=1000):
        root = Node(gameState)
        for i in range(maxSimulation):
            node = root
            newState = copy.deepcopy(gameState)
            while not node.isLeaf() and not newState.noMove(playerID):
                node = node.selectChild()
                newState = copy.deepcopy(node.state)
            if not newState.noMove(playerID):
                node.expand()
                node = node.selectChild()
                newState = node.state
            rank = simulate(newState)
            node.backpropagate(rank)
            if i % 5 == 0:
                for c in range(len(root.children)):
                    root.children[c].children.clear()
        bestMove = root.bestChild().move
        return bestMove



    algorithm = 'minimax'
    state = GameState(mapStat, sheepStat)

    if algorithm == 'minimax':
        return minimax(state, maxDepth, 1, float("-inf"), float("inf"))
    elif algorithm == 'mcts':
        return MCTS(state, 100)
    elif algorithm == 'random':
        legalMoves = state.getLegalMoves(playerID)
        return random.choice(legalMoves) if legalMoves else None
    else:
        my_stat = []
        best_distance = 0
        print(mapStat)
        for r in range(len(mapStat)):
            for c in range(len(mapStat[r])):
                if mapStat[r][c] == playerID:
                    my_stat.append(((r, c), sheepStat[r][c]))
        for stat in my_stat:
            if stat[1] <= 1:
                continue
            for dir_i in range(len(DIRECTION)):
                if dir_i == 4:
                    continue
                dir = DIRECTION[dir_i]
                distance = 0
                end = False
                r, c = stat[0][0], stat[0][1]
                while not end:
                    if (r + dir[0]) >= 0 and (r + dir[0]) <= len(mapStat)-1 and (c + dir[1]) >= 0 and (c + dir[1]) <= len(mapStat[r])-1 \
                        and mapStat[r + dir[0]][c + dir[1]] == 0:
                        distance += 1
                        r += dir[0]
                        c += dir[1]
                    else:
                        end = True
                if distance > best_distance:
                    best_distance = distance
                    step = [stat[0], int(stat[1] // 2), dir_i+1]
        return step


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
