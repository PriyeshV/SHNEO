import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from custom_gridworlds.simple_maze.maze import Maze


def make_undirected(A: NDArray[np.number], method) -> NDArray[np.number]:
    if method not in ['max', 'mean']:
        exit("Invalid method to make graph undirected")
    if method == 'max':
        return np.maximum(A, A.T)
    else:
        return (A + A.T) / 2


def is_symmetric(matrix):
    return int(np.allclose(matrix, matrix.T, atol=1e-8, rtol=1e-5))


def make_stochastic(A: NDArray[np.number], axis=1) -> NDArray[np.number]:
    row_sum = np.sum(A, axis=axis, keepdims=True)
    return A / row_sum


def get_dynamics_tensor(maze: Maze):
    """ Return: Per-Action Adjacency Matrix """
    n_states, n_actions = maze.n_states, len(maze.ACTIONS)

    all_empty_cells = maze.all_empty_grids()
    ij_to_i = {tuple(pos): i for i, pos in enumerate(all_empty_cells)}
    i_to_ij = {v: k for k, v in ij_to_i.items()}

    A_ass = np.zeros((n_actions, n_states, n_states), dtype=bool)
    R_ass = np.zeros((n_actions, n_states, n_states), dtype=np.float32)
    # R_ass = R_ass - 1
    for action_id, action in enumerate(maze.ACTIONS):
        direction = maze.COMPASS[action]
        for state_id in range(n_states):
            state_xy = np.asarray(i_to_ij[state_id])
            is_goal_state = np.array_equal(state_xy, maze.goal)
            next_state = state_xy + direction
            if maze.is_empty(next_state):  # and not is_goal_state:
                next_state_id = ij_to_i[tuple(next_state)]
            else:
                next_state_id = state_id
            A_ass[action_id, state_id, next_state_id] = 1
            is_next_goal_state = np.array_equal(i_to_ij[next_state_id], maze.goal)
            if is_next_goal_state:  # and not is_goal_state:
                R_ass[action_id, state_id, next_state_id] = 1
    # test_matrix(A_ass[0], i_to_ij)
    return A_ass.astype(np.int8), R_ass


# useful for testing
def get_edges_in_xy(A, i_to_ij):
    """ Returns edges in x,y coordinates of the grid"""
    rows, cols = np.where(A)
    rows_xy = [i_to_ij[i] for i in rows]
    cols_xy = [i_to_ij[i] for i in cols]
    edges_xy = list(zip(rows_xy, cols_xy))
    return edges_xy


def reconstruct_adjacency_matrix(SA, n_states, n_actions):
    """ Reconstruct the per-action adjacency matrices R from the state-action graph SA """
    R_re = np.zeros((n_actions, n_states, n_states), dtype=bool)
    for sa_from in range(n_states * n_actions):
        for sa_to in range(n_states * n_actions):
            if SA[sa_from, sa_to]:
                state_from, action_from = sa_from // n_actions, sa_from % n_actions
                state_to = sa_to // n_actions
                R_re[action_from, state_from, state_to] = True

    return R_re


def test_SA_matrix(SA, R, n_states, n_actions):
    R_re = reconstruct_adjacency_matrix(SA, n_states, n_actions)
    if np.array_equal(R, R_re):
        print('The reconstructed R is the same as the original R.')
    else:
        print('The reconstructed R differs from the original R.')


def get_state_graph(A_ass):
    # A_ass in |A|x|S|x|S|
    A_s = np.sum(np.sum(A_ass, axis=2), axis=0).astype(bool)
    return A_s


def get_state_action_graph(A):
    """ STATE-ACTION Graph (|S||A| X |S||A|)
        Probably not useful if inadmissible actions are turned into self-loops
        - all state-action nodes will have an edge
        - all state-action nodes will be connected to the all the actions of the next state
    """
    assert A.ndim == 3
    n_actions, n_states, _ = A.shape
    SA = np.zeros((n_states * n_actions, n_states * n_actions), dtype=bool)
    for state_id in range(n_states):
        for action_id in range(n_actions):
            next_states = np.where(A[action_id, state_id, :])[0]
            for next_state_id in next_states:
                next_actions = np.unique(np.where(A[:, next_state_id, :])[0])
                for next_action_id in next_actions:
                    SA[state_id * n_actions + action_id, next_state_id * n_actions + next_action_id] = 1
    return SA


def get_state_action_P(P_ass):
    assert P_ass.ndim == 3
    n_actions, n_states, _ = P_ass.shape
    n_states_actions = n_states*n_actions
    P_sa = np.zeros((n_states, n_actions, n_states, n_actions))
    for state_id in range(n_states):
        for action_id in range(n_actions):
            next_states = np.where(P_ass[action_id, state_id, :])[0]
            for next_state_id in next_states:
                P_sa[state_id, action_id, next_state_id, :] = P_ass[action_id, state_id, next_state_id] * (1/(2*n_actions))
    return P_sa.reshape(n_states_actions, n_states_actions)


def get_dynamics_matrix(A_ass, state_action, undirected):
    assert A_ass.ndim == 3
    if state_action:
        W = get_state_action_graph(A_ass)
    else:
        W = A_ass.mean(axis=0)
    if undirected:
        W = make_undirected(W, method='max')
    return W.astype(bool)


def plot_elbow_curve(values, k):
    plt.plot(range(1, len(values) + 1), values, 'bo-')
    plt.axvline(x=k, color='r', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title('Elbow Curve')
    plt.show()


def find_k_elbow(values):
    differences = np.diff(values).__abs__()
    k = min(np.argmax(differences) + 2, len(values))
    # plot_elbow_curve(values, k)
    return k


def get_singular_rep(A, k=None, inv=False):
    assert A.ndim == 2
    if k is not None and k <= 0:
        exit('n_dims cannot be <=0 ')
    U, S, Vh = np.linalg.svd(A, full_matrices=True, hermitian=is_symmetric(A))
    V = Vh.T
    sorted_indices = np.argsort(S)
    if inv:
        sorted_indices = sorted_indices[::-1]

    # return top-k values
    if k is None:
        k = find_k_elbow(S[sorted_indices])
    top_k = sorted_indices[:k]
    return U[:, top_k], V[:, top_k], S[top_k]


def get_eigen_rep(A, k=None, inv=False):
    assert A.ndim == 2
    if k is not None and k <= 0:
        exit('n_dims cannot be <=0 ')

    is_sym = np.allclose(A, A.T, atol=1e-8, rtol=1e-5)
    if is_sym:
        S, U = np.linalg.eigh(A)
    else:
        S, U = np.linalg.eig(A)
    sorted_indices = np.argsort(S)
    if inv:
        sorted_indices = sorted_indices[::-1]

    # return top-k values
    if k is None:
        k = find_k_elbow(S[sorted_indices])
    top_k = sorted_indices[:k]
    U, S = U[:, top_k], S[top_k]
    return U, S


def plot_3d_embeddings(embedding, i_to_ij, c='b', title='', annotate=False, R=None, state_action=True, use_ids=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title=title)
    embedding[np.isclose(embedding, 0)] = 0
    if R is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=c, marker='o')
    else:
        R = R.flatten() if state_action else R.mean(axis=1)
        ind = R <= 0
        ax.scatter(embedding[ind, 0], embedding[ind, 1], embedding[ind, 2], c=c, marker='o')
        ind = R > 0
        ax.scatter(embedding[ind, 0], embedding[ind, 1], embedding[ind, 2], c='r', marker='x', s=100)
    if annotate:
        n_actions = 4
        colors = ['r', 'g', 'b', 'm']
        for sa_i in range(embedding.shape[0]):
            if state_action:
                state_id, action_id = sa_i // n_actions, sa_i % n_actions
            else:
                state_id, action_id = sa_i, 0
            if not use_ids:
                ax.text(embedding[sa_i, 0], embedding[sa_i, 1], embedding[sa_i, 2], str(i_to_ij[state_id]), c=colors[action_id])
            else:
                ax.text(embedding[sa_i, 0], embedding[sa_i, 1], embedding[sa_i, 2], str(sa_i), c=colors[action_id])
    plt.show()