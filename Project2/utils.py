from game_state import GameState
import numpy as np

def findConnected(gameState: GameState, id):
    visited = set()
    connectedRegions = []

    def dfs(row, col, region):
        if row < 0 or row >= len(gameState.mapStat) or \
            col < 0 or col >= len(gameState.mapStat[0]) or \
            (row, col) in visited:
            return
        if gameState.mapStat[row, col] == id:
            visited.add((row, col))
            region.append((row, col))
            dfs(row + 1, col, region)
            dfs(row - 1, col, region)
            dfs(row, col + 1, region)
            dfs(row, col - 1, region)

    for row, col in np.ndindex(gameState.mapStat.shape):
        if gameState.mapStat[row, col] == id and (row, col) not in visited:
            region = []
            dfs(row, col, region)
            connectedRegions.append(region)
    return connectedRegions