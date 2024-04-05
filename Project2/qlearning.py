import STcpClient
import numpy as np
import random
import sys

sys.path.append("../..")
from game_state import *
from time import time

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
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
    def apply_policy(game_state: GameState, id, epsilon):
        legalMoves = game_state.getLegalMoves(id)
        if len(legalMoves) == 0:
            #print("this agent has ended")
            #print("mapStat: \n",game_state.mapStat)
            #print("sheep: \n", game_state.sheep)
            # input("Press Enter to continue...")
            return [], True
        highestScore = 0
        
        greedyMoves = []
        best_distance = 0
        for move in legalMoves:
            if move[1] <= 1:
                continue
            for dir_i in range(len(DIRECTION)):
                if dir_i == 4:
                    continue
                dir = DIRECTION[dir_i]
                distance = 0
                end = False
                r, c = move[0][0], move[0][1]
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
                    greedyMoves = [move]
                elif distance == best_distance:
                    greedyMoves.append(move)
        
        if greedyMoves:
            chosenMove = greedyMoves[np.random.choice(range(len(greedyMoves)))]

            if np.random.uniform(0, 1) < epsilon:
                chosenMove = legalMoves[np.random.choice(range(len(legalMoves)))]

            return chosenMove, False
        else:
            return legalMoves[np.random.choice(range(len(legalMoves)))], True
    
    def q_learning(game_state: GameState, n_episodes=1000, alpha=0.05, gamma=0.9):
        #print("start q-learning: ")
        start = time()
        currentMap = game_state.mapStat
        currentSheep = game_state.sheep
        
        for k in range(n_episodes):
            if time() - start > 2.8:
                # print(f'Iterations: {k}')
                break
            episode_end = False
            epsilon = max(1 / (k + 1), 0.1)
            i = 0
            currentID = playerID
            while not episode_end:
                #print(f"time {i} in the while loop, ID = {currentID}")
                i += 1
                # print("in the while loop")
                # action_index = self.agent.apply_policy(state, epsilon)
                """ game_state._calculateScore()
                score0 = game_state.getScore(playerID) """
                chosenMove, episode_end = apply_policy(game_state, currentID, epsilon)
                if episode_end:
                    #print("already no moves available")
                    break
                # print(f"player {current_id}: at {chosenMove[0][0]}, {chosenMove[0][1] }, move: {DIRECTION[chosenMove[-1] - 1]}")
                row0, col0 = chosenMove[0][0], chosenMove[0][1]
                chosenMoveIndex = chosenMove[-1] - 1
                
                new_state = game_state.getNextState(chosenMove, currentID)
                game_state.mapStat = new_state.mapStat
                game_state.sheep = new_state.sheep

                game_state._calculateScore()
                score1 = game_state.getScore(currentID)

                reward = score1 #  - score0
                # game_state._getAgentState()
                episode_end = game_state.gameOver()

                successor_move, episode_end = apply_policy(game_state, currentID, epsilon)

                if not episode_end: 
                    #print("successor move: ",successor_move)
                    # print("successor_move: ", successor_move)
                    row1, col1 = successor_move[0][0], successor_move[0][1]
                    successorMoveIndex = successor_move[-1] - 1

                    action_value = game_state.Q_table[row0, col0, chosenMoveIndex]
                    #print("shapes: ")
                    #print(successorMoveIndex)
                    # print(game_state.Q_table)
                    #print(game_state.Q_table[row0, col0, chosenMoveIndex])
                    #print(reward)
                
                    successor_action_value = game_state.Q_table[row1, col1, successorMoveIndex]
                else:
                    successor_action_value = 0
                    break

                # print(successor_action_value)
                
                Q_table_new_cell = game_state.Q_table[row0, col0, chosenMoveIndex] + \
                            alpha * (reward + gamma * successor_action_value - action_value)
                
                #print("action_value: ", action_value)
                #print("q table new cell: ", Q_table_new_cell)
                game_state.Q_table[row0, col0, chosenMoveIndex] = Q_table_new_cell
                #print("new cell: ",Q_table_new_cell)
                currentID += 1
                if currentID == 5:
                    currentID = 1
                
                #print("learned q-table:", game_state.Q_table)
                # input("Press Enter to continue...")
            game_state.mapStat = currentMap
            game_state.sheep = currentSheep

    algorithm = "qlearning"
    state = GameState(mapStat, sheepStat)
    q_learning(state, 1000, 0.05, 0.9)

    if algorithm == "random":
        legalMoves = state.getLegalMoves(playerID)
        return random.choice(legalMoves) if legalMoves else None
    if algorithm == "qlearning":
        print("mapStat: \n", state.mapStat)
        print("sheep: \n", state.sheep)
        
        legalMoves = state.getLegalMoves(playerID)
        bestMove = None
        highestValue = float('-inf')  # Start with the lowest possible value
        #print(f"player {playerID}'s Q-table")
        #print(state.Q_table)
        for move in legalMoves:
            print("move: ", move)
            row, col = move[0][0], move[0][1]  # Assuming move format is ((row, col), direction)
            moveIndex = move[-1] - 1
            currentValue = state.Q_table[row, col, moveIndex]
            
            if currentValue > highestValue:
                highestValue = currentValue
                bestMove = move

        if not bestMove:
            return [(0,0),0,1]
        print("bestMove: ", bestMove)
        return bestMove


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
