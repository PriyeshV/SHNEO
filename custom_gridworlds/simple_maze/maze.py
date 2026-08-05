import pygame
import numpy as np


class Maze:
    # From the agent's POV
    ACTIONS = ['L', 'R', 'U', 'D']
    action_symbols = ['→', '←', '↓', '↑']
    # Note: For a cell at (row, col):
    COMPASS = {
        'L': (0, -1),  # 'L' = move col - 1
        'R': (0,  1),  # 'R' = move col + 1
        'U': (-1, 0),  # 'U' = move row + 1
        'D': ( 1, 0)   # 'D' = move row - 1
    }

    def __init__(self, maze_factory, screen_size=None, enable_render=False, get_images=False):
        """
        :param maze_factory: An object that provides .get_maze() -> np.array
        :param screen_size: (height, width) of the rendering window.
                            If None, defaults to the maze's (height, width).
        :param enable_render: If True, display the maze via Pygame.
        :param get_images: If True, return an RGB image array from view_update().
        """
        self.__robot = None
        self.__game_over = False
        self.__get_images = get_images
        self.__enable_render = enable_render

        # Build the maze from the factory
        self._build_maze(maze_factory)

        self.screen = None
        if screen_size is None:  # If no screen_size is provided, use the maze dimensions.
            screen_size = self.maze_size
        else:
            # If a single integer is given, use a square window (size, size)
            if isinstance(screen_size, (int, float)):
                screen_size = (int(screen_size), int(screen_size))
            # If a list/tuple/ndarray is given, try to interpret it as (H, W)
            elif isinstance(screen_size, (list, tuple, np.ndarray)):
                if len(screen_size) == 2:
                    screen_size = (int(screen_size[0]), int(screen_size[1]))
                elif len(screen_size) == 1:
                    screen_size = (int(screen_size[0]), int(screen_size[0]))
                else:
                    raise ValueError(f"screen_size must be a scalar or length-2 sequence, got length {len(screen_size)}")
            else:
                raise TypeError(f"screen_size must be int, float, or a length-1/2 sequence, got type {type(screen_size)}")

        self.screen_size = screen_size
        if get_images or enable_render:
            self._setup_screen(self.screen_size)

    def _build_maze(self, maze_factory):
        """
        Retrieves the maze array from the factory and sets up
        basic properties like height, width, entrance, and goal.
        """
        self._maze = maze_factory.get_maze()
        self._height = self._maze.shape[0]  # number of rows
        self._width = self._maze.shape[1]   # number of columns
        self._maze_str = maze_factory.maze_str

        o_positions = np.argwhere(self._maze == 'O')
        if o_positions.size > 0:
            # record entrance at the marker
            start_rc = tuple(map(int, o_positions[0]))
            # IMPORTANT: treat 'O' as empty for all downstream logic
            self._maze[start_rc] = ' '
        else:
            start_rc = None

        # Identify all empty grid cells
        self._all_empty_grids = np.argwhere(self._maze == ' ')
        self._n_states = self._all_empty_grids.shape[0]

        # Build a position->index mapping for "one-hot" style or tabular indexing
        self._pos_indices = {}
        for i, pos in enumerate(self._all_empty_grids):
            self._pos_indices[tuple(pos)] = i

        # Maze size as [height, width] for reference
        self.maze_size = (self._height, self._width)

        # Define entrance and goal
        if start_rc is not None:
            self.__entrance = np.array(start_rc, dtype=int)
        else:
            self.__entrance = self.first_empty_grid()
        self.__goal = self.last_empty_grid()
        self.__robot = self.__entrance.copy()

    def _setup_screen(self, screen_size):
        """  Initialize Pygame, set the window size, and fill the screen background.  """
        pygame.init()
        pygame.display.set_caption(self._maze_str)
        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(screen_size)
        self.screen.fill((255, 255, 255))

    def quit_game(self):
        """  Gracefully close the game window and quit Pygame.  """
        try:
            self.__game_over = True
            if self.__enable_render:
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def reset_robot(self, position):
        self.__robot = np.array(position, dtype=int).copy()  # Move the robot to a specified position in the maze.

    def set_entrance(self, position):
        self.__entrance = np.array(position, dtype=int).copy()  # Redefine the maze's entrance (e.g., for random starts)

    def move_robot(self, direction):
        """
        Attempt to move the robot in the given direction (one of ACTIONS).
        If the destination cell is open, the robot moves; otherwise, it stays put.
        """
        if direction not in self.ACTIONS:
            raise ValueError(f"{direction} is not a valid action")

        if self.is_open(self.__robot, direction):
            delta = self.COMPASS[direction]  # (d_row, d_col)
            self.__robot += np.array(delta)

    def is_open(self, cell_id, direction):
        """ Check if the cell in the given direction from cell_id is empty (' '). """
        delta = self.COMPASS[direction]
        new_pos = cell_id + delta
        return self.is_empty(new_pos)

    def __draw_walls(self, surface=None, colour=(0, 0, 0), transparency=255):
        """ Draw all non-empty cells (e.g., walls) in the given colour. """
        walls = np.argwhere(self._maze != ' ')
        for (r, c) in walls:
            self.__colour_cell([r, c], colour=colour, transparency=transparency, surface=surface)
        walls = np.argwhere(self._maze != ' ')
        for (r, c) in walls:
            self.__colour_cell([r, c], colour=colour, transparency=transparency, surface=surface)

    def __draw_robot(self, surface=None, colour=(0, 150, 0), transparency=255):
        self.__colour_cell(self.robot, colour=colour, transparency=transparency, surface=surface)  # Draw the robot in green.

    def __draw_entrance(self, surface=None, colour=(0, 0, 150), transparency=255):
        self.__colour_cell(self.entrance, colour=colour, transparency=transparency, surface=surface)  # Draw the entrance cell in blue.

    def __draw_goal(self, surface=None, colour=(150, 0, 0), transparency=255):
        self.__colour_cell(self.goal, colour=colour, transparency=transparency, surface=surface)  # Draw the goal cell in red.

    def __colour_cell(self, cell, colour, transparency, surface=None):
        """ Convert (row, col) to (x, y) pixel coordinates, then draw a rectangle. """
        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must be a tuple, list, or numpy array of size 2")

        if surface is None:
            surface = self.screen

        # cell_w, cell_h = self.CELL_W, self.CELL_H
        surf_width, surf_height = surface.get_size()
        cell_w, cell_h = surf_width / self.width, surf_height / self.height

        # cell[0] = row => y, # cell[1] = col => x
        row, col = cell[0], cell[1]
        x = int(row * cell_h)
        y = int(col * cell_w)
        pygame.draw.rect(surface, colour + (transparency,), (x, y, int(cell_w), int(cell_h)))

    def view_update(self):
        """  Public method to refresh the display and optionally return an RGB image array.  """
        if self.__enable_render or self.__get_images:
            return self.__view_update()

    def __view_update(self):
        """  Internal method that handles the actual drawing steps and returns an image if needed.  """
        if self.screen is not None and not self.__game_over:
            # Clear screen
            native_surface = pygame.Surface(self.maze_size, flags=pygame.SRCALPHA)
            native_surface.fill((255, 255, 255))

            # Draw items
            self.__draw_walls(surface=native_surface)
            self.__draw_goal(surface=native_surface)
            self.__draw_robot(surface=native_surface)

            if native_surface.get_size() != self.screen_size:
                final_surface = pygame.transform.smoothscale(native_surface, self.screen_size)
            else:
                final_surface = native_surface

            # Refresh the display
            self.screen.blit(final_surface, (0, 0))
            if self.__enable_render:
                pygame.display.flip()
                pygame.time.delay(100)

            # If get_images is True, return the RGB array from the surface
            if self.__get_images:
                return pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8)

    def __controller_update(self):
        """  Process Pygame events (like window close).  """
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def is_empty(self, pos):
        """  Check if the given (row, col) is within bounds and if that cell is ' '.  """
        r, c = pos
        if 0 <= r < self._height and 0 <= c < self._width:
            return self._maze[r, c] == ' '
        else:
            return False

    def pos_index(self, pos):
        return self._pos_indices[tuple(pos)]  # Returns the one-hot/tabular index for the given position if needed.

    def all_empty_grids(self):
        return np.argwhere(self._maze == ' ')  # Return a list of all (row, col) positions where the maze cell is ' '.

    def random_empty_grids(self, k):
        """  Return k random empty positions in the maze.  """
        empty_grids = self.all_empty_grids()
        selected = np.random.choice(len(empty_grids), size=k, replace=False)
        return empty_grids[selected]

    def first_empty_grid(self):
        """  Return the top-leftmost empty cell as the entrance.  """
        empty_grids = self.all_empty_grids()
        assert empty_grids.shape[0] > 0
        # Sort by row then col
        empty_grids = empty_grids[np.lexsort((empty_grids[:, 1], empty_grids[:, 0]))]
        return empty_grids[0]

    def last_empty_grid(self):
        """  Return the bottom-rightmost empty cell as the goal.  """
        empty_grids = self.all_empty_grids()
        assert empty_grids.shape[0] > 0
        # Already sorted or you can just take [-1] from unsorted
        return empty_grids[-1]

    def __getitem__(self, key):
        """  Allows direct indexing into the underlying maze array.  """
        return self._maze[key]

    def __setitem__(self, key, val):
        """  Allows direct setting in the underlying maze array.  """
        self._maze[key] = val

    # ----------------------------
    # Properties
    # ----------------------------
    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def n_states(self):
        return self._n_states

    @property
    def maze(self):
        return self._maze

    @property
    def robot(self):
        return self.__robot.copy()

    @property
    def entrance(self):
        return self.__entrance.copy()

    @property
    def goal(self):
        return self.__goal.copy()

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return self.__screen_size

    @property
    def SCREEN_W(self):
        return self.SCREEN_SIZE[1]  # Returns the width in pixels of the Pygame screen.

    @property
    def SCREEN_H(self):
        return self.SCREEN_SIZE[0]  # Returns the height in pixels of the Pygame screen.

    @property
    def CELL_W(self):
        return float(self.SCREEN_W/self.width)  # Pixel width of each cell = screen width / maze width

    @property
    def CELL_H(self):
        return float(self.SCREEN_H/self.height)  # Pixel height of each cell = screen height / maze height
