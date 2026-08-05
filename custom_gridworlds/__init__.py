from gymnasium.envs.registration import register

all_mazes = ['GridMaze-7', 'GridMaze-9', 'GridMaze-11', 'HardMaze', 'GridMaze-17', 'GridMaze-19', 'GridMaze-26',
             'GridMaze-32', 'GridRoom-1', 'GridRoom-4', 'GridRoom-16', 'GridRoom-32', 'GridRoom-64',
             'OneRoom', 'TwoRoom', 'FourRoom', 'Small-GridRoom-4', 'GridRoom-4_1', 'GridRoom-4_2', 'GridRoom-9']


n_states = {'GridMaze-7': 11, 'GridMaze-9': 13, 'GridMaze-11': 17, 'GridMaze-17': 87,
            'GridMaze-19': 161, 'GridMaze-26': 388, 'GridMaze-32': 476,
            'GridRoom-1': 225, 'GridRoom-4': 104, 'GridRoom-16': 271, 'GridRoom-32': 544, 'GridRoom-64': 1088,
            'OneRoom': 225, 'TwoRoom': 212, 'FourRoom': 204, 'Small-GridRoom-4': 28, 'HardMaze': 148,
            'GridRoom-9': 156}

ACTIONS = ['L', 'R', 'U', 'D']

register(
    id='Simple_Maze-v0',
    entry_point='custom_gridworlds.simple_maze:MazeEnv',
    max_episode_steps=50,
    kwargs={}
)
