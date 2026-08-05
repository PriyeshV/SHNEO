import sys
import gymnasium as gym

sys.path.append('/')
from simple_maze.scripts.utils import *


if __name__ == "__main__":
    render_mode = None  # For testing, set render_mode='human
    for maze_type in all_mazes:
        # maze_type = 'GridMaze-7'
        env = gym.make('Simple_Maze-v0', maze_type=maze_type, obs_mode='xy', random_start=False,
                       render_mode=render_mode, render_fps=20)

        # Compute per-action adjacency matrices:
        R = get_adjacency_matrix(env.unwrapped.maze)
        n_states, n_actions = np.shape(R)[1], len(ACTIONS)

        # Compute per-action transition matrices:
        #   Note: P = R, for 1-regular graphs
        P_R = R.astype(np.float32)
        for action_id in range(len(ACTIONS)):
            P_R[action_id, :, :] = make_stochastic(R[action_id, :, :], axis=1)  # unnormalized P = D^-1A

        # UNDIRECTED GRAPH
        # Compute per-action transition matrices for undirected action graphs:
        #   Note: - per-action graph is no-more a regular graph when you make it undirected
        #           because of inadmissible self-loops
        #         - Transition matrices of an undirected graph need not be symmetric, P_s != P_s.T
        #         - To obtain a symmetric matrices,
        #           (a) get the laplacian, L = I - D^-1A
        #           (b) get the symmetric/normalized transition matrices, P = D^{-1/2}AD^{-1/2}
        R_und = np.zeros_like(R)
        for action_id in range(len(ACTIONS)):
            R_und[action_id, :, :] = make_undirected(R[action_id, :, :], method='max')

        P_R_und = np.zeros_like(P_R)
        for action_id in range(len(ACTIONS)):
            P_R_und[action_id, :, :] = make_stochastic(R_und[action_id, :, :], axis=1)

        # Reduction: Tensor -> Matrix:

        # OPTION 1: STATE x STATE Graph
        W = np.mean(R, axis=0)
        P = make_stochastic(W, axis=1)
        W_und = make_undirected(W, method='max')  # should try mean
        P_und = make_stochastic(W_und, axis=1)
        L_und = np.eye(W_und.shape[0]) - P_und

        # OPTION 2: STATE-ACTION x STATE-ACTION Graph
        SA = get_state_action_graph(R)
        P_SA = make_stochastic(SA, axis=1)
        SA_und = make_undirected(SA, method='max')
        P_SA_sym = make_stochastic(SA_und, axis=1)
        L_SA_und = np.eye(SA_und.shape[0]) - P_SA

        # Singular System
        U, s, Vh = np.linalg.svd(P)
        U, s, Vh = np.linalg.svd(P_SA)

        # Eigen System
        U, s = np.linalg.eig(L_und)
        U, s = np.linalg.eig(L_SA_und)
