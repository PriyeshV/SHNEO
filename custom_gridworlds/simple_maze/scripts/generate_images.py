import os
import sys
from pathlib import Path

import cv2

sys.path.append('/')

import numpy as np
import custom_gridworlds
import gymnasium as gym
from matplotlib import pyplot as plt

custom_mazes = ['GridMaze-7', 'GridMaze-9', 'GridMaze-11', 'HardMaze', 'GridMaze-17', 'GridMaze-19', 'GridMaze-26',
                'GridMaze-32', 'GridRoom-1', 'GridRoom-4', 'GridRoom-16', 'GridRoom-32', 'GridRoom-64',
                'OneRoom', 'TwoRoom', 'FourRoom', 'Small-GridRoom-4']
# custom_mazes = ['GridRoom-16']


def visualize_array(img_array, maze_name):
    plt.imshow(img_array)
    plt.title(maze_name)
    plt.tight_layout()  # Minimizes padding around the image
    plt.show()


render_mode = None  # For testing, set render_mode='human
# image_size = (32, 32)
# image_size = (64, 64)
image_size = None
for maze_type in custom_mazes:
    env = gym.make('Simple_Maze-v0', maze_type=maze_type, obs_mode='xy', random_start=False,
                   render_mode=render_mode, render_fps=20, screen_size=image_size)
    maze = env.unwrapped.maze
    # env = ResizeObservation(env, 32)   #if image_size is not None else env
    (height, width) = maze.height, maze.width
    (img_h, img_w) = (image_size[0], image_size[1]) if image_size is not None else (height, width)
    print('Maze: ', maze_type, '[ Height: ', height, ', Width: ', width, ']')

    print('----------------------------------------------------------------')
    empty_grids = maze.all_empty_grids()
    state_images = np.zeros((height, width, img_h, img_w, 3), dtype=np.uint8)
    for step, (i, j) in enumerate(empty_grids):
        maze.reset_robot(np.array([i, j], dtype=int))
        img = maze.view_update()
        state_images[i, j, :, :, :] = img
        # visualize_array(img, maze_type)
    directory = Path(__file__).parent.parent
    dir_suffix = '_'+str(image_size[0]) if image_size is not None else ''
    np.save(os.path.join(directory, 'mazes/images'+dir_suffix, maze_type), state_images)
    print('Done')
    print('--------------------------------')