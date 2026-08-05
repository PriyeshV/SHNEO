import os
import numpy as np
from pathlib import Path


class FactoryBase:
    _maze = None

    def get_maze(self):
        return self._maze


class CustomMaze(FactoryBase):
    # Environments are taken from here: https://github.com/tarod13/laplacian_dual_dynamics/tree/main/src/env/grid/txts
    def __init__(self, maze_str='GridRoom-4'):
        self.maze_str = maze_str
        self._parse_txt(maze_str)

    def _parse_txt(self, filename):
        directory = Path(__file__).parent
        filepath = os.path.join(directory, 'mazes/txts/', filename+'.txt')
        file = open(filepath, 'r')
        self._parse_maze(file)

    def _parse_maze(self, maze_source):
        width = 0
        height = 0
        maze_matrix = []
        for row in maze_source:
            row_vector = row.strip()
            maze_matrix.append(row_vector)
            height += 1
            width = max(width, len(row_vector))
        maze_array = np.zeros([height, width], dtype=str)
        maze_array[:] = ' '
        for i, row in enumerate(maze_matrix):
            for j, val in enumerate(row):
                maze_array[i, j] = val

        self._maze = np.asarray(maze_array)

class SquareRoom(FactoryBase):
    """generate a square room with given size"""

    def __init__(self, size):
        self.maze_str = 'OneRoom'
        size = 15 if size is None else size
        maze_array = np.zeros([size + 2, size + 2], dtype=str)
        maze_array[:] = ' '
        maze_array[0] = '-'
        maze_array[-1] = '-'
        maze_array[:, 0] = '|'
        maze_array[:, -1] = '|'
        maze_array[0, 0] = '+'
        maze_array[0, -1] = '+'
        maze_array[-1, 0] = '+'
        maze_array[-1, -1] = '+'
        self._maze = maze_array


class FourRooms(FactoryBase):
    """generate four rooms, each with the given size"""

    def __init__(self, size):
        self.maze_str = 'FourRoom'
        size = 7 if size is None else size
        maze_array = np.zeros([size * 2 + 3, size * 2 + 3], dtype=str)
        maze_array[:] = ' '
        wall_idx = [0, size + 1, size * 2 + 2]
        maze_array[wall_idx] = '-'
        maze_array[:, wall_idx] = '|'
        maze_array[wall_idx][:, wall_idx] = '+'
        door_idx = [int((size + 1) / 2), int((size + 1) / 2) + 1,
                    int((size + 1) / 2) + size + 1, int((size + 1) / 2) + size + 2]
        maze_array[size + 1, door_idx] = ' '
        maze_array[door_idx, size + 1] = ' '
        self._maze = maze_array


class TwoRooms(FactoryBase):
    def __init__(self, size):
        self.maze_str = 'TwoRoom'
        size = 15 if size is None else size
        maze_array = np.zeros([size + 2, size + 2], dtype=str)
        maze_array[:] = ' '
        hwall_idx = [0, int((size + 1) / 2), size + 1]
        vwall_idx = [0, size + 1]
        maze_array[hwall_idx] = '-'
        maze_array[:, vwall_idx] = '|'
        maze_array[hwall_idx][:, vwall_idx] = '+'
        door_idx = [int((size + 1) / 2), int((size + 1) / 2) + 1]
        maze_array[hwall_idx[1], door_idx] = ' '
        self._maze = maze_array