import sys
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('/')
from simple_maze.scripts.utils import *


def get_networkx_visualization(adj_mat, title=''):
    import networkx as nx
    # np.fill_diagonal(adj_mat, 0)
    G = nx.DiGraph(adj_mat)
    # pos = nx.spectral_layout(G)
    # nx.draw(G, pos, with_labels=True, node_color='skyblue', font_size=10)
    nx.draw_kamada_kawai(G)
    plt.title(title)
    plt.show()


def visualize_vectors(U, S, title='', small=True, k=3, annotate=False):
    # Get the indices of the top-k singular values
    if not small:
        top_k_indices = np.argsort(S)[-k:][::-1]
    else:
        top_k_indices = np.argsort(S)[:k]

    # Select the top-k singular values and corresponding vectors
    plot_3d_embeddings(U[:, top_k_indices], title=title, annotate=annotate)


if __name__ == "__main__":
    render_mode = None  # For testing, set render_mode='human
    for maze_type in all_mazes:
        # maze_type = 'FourRoom'
        maze_type = 'GridMaze-7'
        env = gym.make('Simple_Maze-v0', maze_type=maze_type, obs_mode='xy', random_start=False,
                       render_mode=render_mode, render_fps=20)
        maze = env.unwrapped.maze
        all_empty_cells = maze.all_empty_grids()
        ij_to_i = {tuple(pos): i for i, pos in enumerate(all_empty_cells)}
        i_to_ij = {v: k for k, v in ij_to_i.items()}

        # Compute per-action adjacency matrices:
        R = get_dynamics_tensor(maze)
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
        # get_spectral_visualization(W, title='State Graph')
        P = make_stochastic(W, axis=1)
        W_und = make_undirected(W, method='max')  # should try mean
        P_und = make_stochastic(W_und, axis=1)
        L_und = np.eye(W_und.shape[0]) - P_und

        # OPTION 2: STATE-ACTION x STATE-ACTION Graph
        SA = get_state_action_graph(R)
        # get_spectral_visualization(SA, title='State Action Graph')
        P_SA = make_stochastic(SA, axis=1)
        SA_und = make_undirected(SA, method='max')
        P_SA_und = make_stochastic(SA_und, axis=1)
        L_SA_und = np.eye(SA_und.shape[0]) - P_SA_und

        # exit()
        # Singular System
        U, s, _ = np.linalg.svd(P)
        visualize_vectors(U, s, small=False, title='Singular: State Graph')
        U, s, _ = np.linalg.svd(P_SA)
        visualize_vectors(U, s, small=False, title='Singular P: State Action Graph', annotate=True)
        U, s, Vh = np.linalg.svd(L_SA_und)
        # This has something interesting
        visualize_vectors(U, s, small=True, title='Singular L_U: State Action Graph', annotate=True)
        visualize_vectors(Vh.T, s, small=True, title='Singular L_V: State Action Graph', annotate=True)

        # Eigen System
        s, U = np.linalg.eig(L_und)
        visualize_vectors(U, s, small=True, title='Eigen: State Graph')
        s, U = np.linalg.eig(L_SA_und)
        visualize_vectors(U, s, small=True, title='Eigen: State Action Graph', annotate=True)
        exit()