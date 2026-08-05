import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from custom_gridworlds import all_mazes
from custom_gridworlds.simple_maze.maze import Maze
from custom_gridworlds.simple_maze.maze_generator import *


def get_images(maze_type, screen_size):
    directory = Path(__file__).parent

    dir_suffix = '_'+str(screen_size) if screen_size is not None else ''
    file_path = os.path.join(directory, 'mazes/images'+dir_suffix, maze_type + '.npy')
    if os.path.exists(file_path):
        print('Pixel data Pre-loaded')
        images = np.load(file_path)
        return images
    else:
        answer = input('No saved image data found. Do you want to render on the go, Y or N ?').strip().lower()
        if answer != 'y':
            exit('Ok. Run scripts/generate_images.py to generate and save images. Exiting now!')
        return


class MazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'],
                'obs_modes': ['one-hot', 'xy', 'pixels'],
                'render_fps': 30}
    room_factories = {'OneRoom': SquareRoom, 'TwoRoom': TwoRooms, 'FourRoom': FourRooms}
    custom_mazes = ['GridMaze-7', 'GridMaze-9', 'GridMaze-11', 'HardMaze', 'GridMaze-17', 'GridMaze-19', 'GridMaze-26',
                    'GridMaze-32', 'GridRoom-1', 'GridRoom-4', 'GridRoom-9', 'GridRoom-16', 'GridRoom-32', 'GridRoom-64',
                    'Small-GridRoom-4', 'GridRoom-4_1', 'GridRoom-4_2']
    assert set(list(room_factories.keys()) + custom_mazes) == set(all_mazes)

    def __init__(self, maze_type='OneRoom', maze_size=None, obs_mode='one-hot', render_mode='rgb_array',
                 render_fps=None, random_start=False, screen_size=None):  # screen_size: None for maze-size images
        assert maze_type in all_mazes, \
            'Custom Grid world: '+maze_type+' not available.\n Available Options: '+all_mazes.__str__()
        self.maze_type = maze_type
        self.enable_render = render_mode == 'human'

        assert obs_mode in self.metadata['obs_modes']
        self.images = get_images(maze_type, screen_size) if obs_mode == 'pixels' else None

        if render_fps is not None:
            self.metadata['render_fps'] = render_fps

        # Select the appropriate factory based on the maze_type
        params = {'enable_render': self.enable_render, 'get_images': self.images is None,
                  'screen_size': screen_size if screen_size is not None else None}
        if maze_type in self.room_factories:
            self.maze = Maze(self.room_factories[maze_type](size=maze_size), **params)
        else:
            self.maze = Maze(CustomMaze(maze_str=maze_type), **params)

        # Set Environment spaces
        self.action_space = spaces.Discrete(len(self.maze.ACTIONS))

        self.obs_mode = obs_mode
        if self.obs_mode == 'xy':  # The first and the last row and column are made up of walls
            low, high = self.normalize_xy([0, 0]), self.normalize_xy(self.maze.maze_size)
            self.observation_space = spaces.Box(low, high, dtype=np.float64)
        elif self.obs_mode == 'one-hot':
            self.observation_space = spaces.MultiBinary(self.maze.n_states)  # All empty cells
        else:  # pixels
            if screen_size is None:
                screen_size = self.maze.maze_size
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_size, screen_size, 3), dtype=np.uint8)

        self.state, self.done = None, False
        self.steps_beyond_done = None
        self.random_start = random_start

        print(f"[MazeEnv.__init__] maze='{self.maze_type}' random_start={self.random_start} "
              f"entrance={self.maze.entrance.tolist()} goal={self.maze.goal.tolist()}")

    def normalize_xy(self, position):
        pos_0 = position[0] / self.maze.maze_size[0] - 0.5
        pos_1 = position[1] / self.maze.maze_size[1] - 0.5
        return np.asarray([pos_0, pos_1], np.float64)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state(self, position):
        if self.obs_mode == 'pixels':
            if self.images is not None:
                state = self.images[self.maze.robot[0], self.maze.robot[1], :, :, :]
            else:
                state = self.maze.view_update()
        elif self.obs_mode == 'one-hot':
            id = self.maze.pos_index(pos=[position[0], position[1]])  # tabular-id
            state = np.eye(self.maze.n_states)[id].astype(dtype=int)
        else:
            state = self.normalize_xy(position)
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            state = np.asarray(self.maze.random_empty_grids(1)[0], dtype=int)
            self.maze.set_entrance(state)
        else:
            # start at explicit 'O' if provided, else first empty (set in Maze._build_maze)
            state = np.asarray(self.maze.entrance, dtype=int)
        self.maze.reset_robot(state)
        self.state = self.get_state(position=state)

        info = {}
        self.done = False
        self.steps_beyond_done = None
        return self.state, info

    def step(self, action):
        if isinstance(action, np.ndarray) and np.ndim(action) > 0:
            self.maze.move_robot(self.maze.ACTIONS[action[0]])
        else:
            self.maze.move_robot(self.maze.ACTIONS[action])

        # compute next observation from the *current* state
        self.state = self.get_state(self.maze.robot)

        reached = np.array_equal(self.maze.robot, self.maze.goal)
        if reached:
            reward, terminated = 1, True
        else:
            reward, terminated = 0, False
            # reward, terminated = -1, False

        info = {}
        # IMPORTANT: do NOT call self.reset() here; let the caller reset after seeing terminated=True
        return self.state, reward, terminated, False, info

    def render(self):
        self.maze.view_update()

    def close(self):
        self.maze.quit_game()