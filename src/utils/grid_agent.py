import gymnasium as gym
import matplotlib.pyplot as plt

from src.utils.utils import *
from sklearn.preprocessing import PolynomialFeatures
from custom_gridworlds.simple_maze.scripts.utils import (get_dynamics_tensor, get_dynamics_matrix,
                                                         get_singular_rep, get_eigen_rep, plot_3d_embeddings)


class World:
    def __init__(self, maze, params):
        # Get Env Dynamics
        self.A_ass, self.R_ass = get_dynamics_tensor(maze)
        n_states, n_actions = np.shape(self.A_ass)[1], len(maze.ACTIONS)

        self.P_ass = np.zeros_like(self.A_ass, dtype=np.float32)
        self.R_sa = np.zeros((n_states, n_actions))
        for action_id in range(n_actions):
            self.P_ass[action_id, :, :] = get_transition_matrix(self.A_ass[action_id, :, :],
                                                                normalized=params.normalized_P)
            self.R_sa[:, action_id] = np.sum(self.R_ass[action_id, :, :] * self.P_ass[action_id, :, :], axis=-1)
        self.R_s = self.R_sa.sum(axis=-1) / n_actions

        # Reduce multi-relational graph to a simple graph
        if params.state_action is not None:
            self.A = get_dynamics_matrix(self.A_ass, state_action=params.state_action, undirected=params.undirected)
            self.P = get_transition_matrix(self.A, normalized=params.normalized_P)
            # if params['state_action']:
            #     self.P = get_state_action_P(self.P_ass)  # this is not a stochastic matrix. This is P_r
            self.n_nodes = self.P.shape[0]
            if params.laplacian:
                self.L = get_rw_laplacian(A=None, P=self.P, normalized=params['normalized_P'])
                # self.L = get_laplacian(A=self.A)

        self.params = params


class GridAgent:
    def __init__(self, params):
        self.params = params
        self.maze_type = params.env.maze_type
        self.gamma = params.env.gamma
        self.env = gym.make('Simple_Maze-v0', maze_type=self.maze_type, obs_mode=params.env.obs_mode,
                            random_start=params.env.rand_reset, render_mode=None, render_fps=20,
                            screen_size=params.env.screen_size)
        maze = self.env.unwrapped.maze
        self.action_symbols = maze.action_symbols
        self.width, self.height = maze.width, maze.height
        self.entrance, self.goal = maze.entrance, maze.goal
        self.max_steps = self.width*self.height

        # Maze states
        all_empty_cells = maze.all_empty_grids()
        self.ij_to_i = {tuple(pos): i for i, pos in enumerate(all_empty_cells)}
        self.i_to_ij = {v: k for k, v in self.ij_to_i.items()}
        self.n_states, self.n_actions = len(self.ij_to_i), len(maze.ACTIONS)

        # World Graph
        self.graph = World(maze, params.graph)

    def compute_P_pi_ss(self, pi):
        P_pi_sas = self.get_P_pi_sas(pi)
        P_pi_ss = P_pi_sas.sum(axis=1)
        assert np.all(P_pi_ss.sum(axis=1) == 1)
        return P_pi_ss

    def get_P_pi_sas(self, pi):
        # \sum_a \sum_s' P_pi(s, a, s') = 1
        P_pi_sas = self.graph.P_ass.transpose(1, 0, 2) * np.expand_dims(pi, axis=2)
        return P_pi_sas

    def get_P_pi_sa_matrix(self, pi,):
        P_pi_sasa = np.zeros((self.n_states, self.n_actions, self.n_states, self.n_actions))
        for s_prime in range(self.n_states):
            for a_prime in range(self.n_actions):
                P_pi_sasa[:, :, s_prime, a_prime] = self.graph.P_ass[:, :, s_prime].transpose(1, 0) * pi[s_prime, a_prime]
        return P_pi_sasa

    def get_P_pi_s_pis_matrix(self, pi,):
        P_pi_sasa = np.zeros((self.n_states, self.n_actions, self.n_states, self.n_actions))
        for s_id in range(self.n_states):
            for a_id in range(self.n_actions):
                next_states = np.where(self.graph.P_ass[a_id, s_id, :])[0]
                for n_s_id in next_states:
                    for n_a_id in range(self.n_actions):
                        P_pi_sasa[s_id, a_id, n_s_id, n_a_id] = pi[s_id, a_id] * self.graph.P_ass[a_id, s_id, n_s_id] * pi[n_s_id, n_a_id]
        return P_pi_sasa

    def get_returns_at_test(self, pi, num_episodes=20):
        env = gym.make('Simple_Maze-v0', maze_type=self.params.env.maze_type, obs_mode='xy',
                       random_start=self.params.env.eval_rand_reset, render_mode=None, render_fps=20)

        returns, steps = [], []
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            t, total_reward = 0, 0
            while not done:
                action = np.argmax(pi[self.ij_to_i[tuple(env.unwrapped.maze.robot)], :])
                next_state, reward, done, _, _ = env.step(action)
                total_reward += (self.gamma ** t) * reward
                t += 1
                if t > self.max_steps:
                    break
            steps.append(t - 1)
            returns.append(total_reward)
        return np.mean(returns), np.mean(steps)

    def visualize_grid(self, policy=None, title='', visual=False):
        plt.clf()
        if policy is not None:
            colors = ['r', 'g', 'b', 'm']
            actions = np.argmax(policy, axis=1)

        # Create a figure and axis
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(self.width))
        ax.set_yticks(np.arange(self.height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.ij_to_i.keys():
                    ax.text(j, i, '#', ha='center', va='center', fontsize=20)
                else:
                    if (i, j) == (self.goal[0], self.goal[1]):
                        ax.text(j, i, 'G', ha='center', va='center', fontsize=20)
                    elif (i, j) == (self.entrance[0], self.entrance[1]):
                        ax.text(j, i, 'E', ha='center', va='center', fontsize=20)
                    if policy is not None:
                        action_id = actions[self.ij_to_i[(i, j)]]
                        ax.text(j, i, self.action_symbols[action_id], c=colors[action_id], ha='center', va='center', fontsize=20)
                    else:
                        ax.text(j, i, self.ij_to_i[(i, j)], c='b', ha='center', va='center', fontsize=20)
        ax.yaxis.set_inverted(True)
        plt.title(title)
        return plt

    def explain_P_ss(self, P_ss):
        assert P_ss.shape[0] == self.n_states
        for i in range(self.n_states):
            neigh_ids = np.where(P_ss[i, :])[0]
            neigh_xy = [self.i_to_ij[neigh_id] for neigh_id in neigh_ids]
            print(self.i_to_ij[i], neigh_xy)


class FA_GridAgent(GridAgent):
    def __init__(self, params):
        super(FA_GridAgent, self).__init__(params)
        if self.params.features.p_dims is None or self.params.features.p_dims <= 0:
            self.k = self.params.features.n_dims
        else:
            self.k = int(self.params.features.p_dims*self.n_states)
        graph = self.graph.L if self.graph.params.laplacian else self.graph.P
        if self.params.features.decomp == 'SVD':
            phi_u, phi_v, lambdas = get_singular_rep(graph, inv=not self.graph.params.laplacian, k=self.k)
            vectors = {'L': phi_u, 'R': phi_v, 'LR': np.concatenate([phi_u, phi_v], axis=1)}
            self.phi = vectors[self.params.features.vector]
            self.lambdas = lambdas if self.params.features.vector != 'LR' else np.tile(lambdas, 2)
            del phi_u, phi_v
        else:
            self.phi, self.lambdas = get_eigen_rep(graph, inv=not self.graph.params.laplacian, k=self.k)
            self.phi, self.lambdas = np.real(self.phi), np.real(self.lambdas)

        # assert len(np.where(self.phi.sum(axis=1) == 0)[0]), 'Phi has zero vectors'
        self.n_dims = self.phi.shape[1]
        self.lambdas = np.append(self.lambdas, [1]) if self.params.features.bias else self.lambdas
        if params.visual.on:
            plot_3d_embeddings(self.phi, self.i_to_ij,
                           title=self.params.features.decomp+': ' + self.params.features.rep,
                           R=self.graph.R_sa, state_action=self.graph.params.state_action,
                           annotate=self.params.visual.emb_annot)

        self.pi_r = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.poly = PolynomialFeatures(degree=self.params.features.poly_deg,
                                       include_bias=self.params.features.bias)

    # def compute_policy_rep_sa(self, pi, T=0):
    #     assert T >= 0
    #     if T == 0:
    #         return self.phi * np.expand_dims(pi, 2)
    #
    #     psi = self.phi.copy()
    #     for t in range(1, T+1):
    #         n_psi_sa = np.zeros_like(psi)
    #         for s_id in range(self.n_states):
    #             for a_id in range(self.n_actions):
    #                 next_states = np.where(self.graph.P_ass[a_id, s_id, :])[0]
    #                 for n_s_id in next_states:
    #                     for n_a_id in range(self.n_actions):
    #                         n_psi_sa[s_id, a_id, :] += (pi[n_s_id, n_a_id] * psi[n_s_id, n_a_id, :])
    #         psi = self.phi + self.gamma*n_psi_sa
    #     return psi

    def compute_policy_rep_sa(self, pi, T=0):
        assert T >= 0

        # T = 0: just policy-weighted features (your original behavior)
        if T == 0:
            return self.phi * np.expand_dims(pi, 2)  # [S, A, d]

        psi = self.phi.copy()  # [S, A, d]

        for _ in range(T):
            n_psi_sa = np.zeros_like(psi)  # [S, A, d]

            # 1) Expected successor feature per *next state* s':
            #    exp_psi[s'] = sum_a' pi[s', a'] * psi[s', a', :]
            exp_psi = (pi[..., None] * psi).sum(axis=1)  # [S, d]

            # 2) For each (s,a), apply transition P(s'|s,a)
            for s_id in range(self.n_states):
                for a_id in range(self.n_actions):
                    # p_row: [S] where p_row[n_s_id] = P(s' = n_s_id | s, a)
                    p_row = self.graph.P_ass[a_id, s_id, :]  # [S]
                    # Dot over next states -> [d]
                    n_psi_sa[s_id, a_id, :] = p_row @ exp_psi  # [d]

            # Bellman-like update
            psi = self.phi + self.gamma * n_psi_sa

        return psi

    def compute_policy_rep_s(self, pi, T=1):
        assert T > 0
        psi = self.phi.copy()
        for t in range(1, T+1):
            n_psi_s = np.zeros_like(psi)
            for s_id in range(self.n_states):
                for a_id in range(self.n_actions):
                    # next_states = np.where(self.graph.P_ass[a_id, s_id, :])[0]
                    # for n_s_id in next_states:
                    #     n_psi_s[s_id, :] += (pi[s_id, a_id] * psi[n_s_id, :])
                    p_row = self.graph.P_ass[a_id, s_id, :]  # shape [n_states]
                    n_psi_s[s_id, :] += pi[s_id, a_id] * (p_row @ psi)  # matrix–vector over states
            psi = self.phi + self.gamma*n_psi_s
        return psi

    def get_policy_rep_exact(self, pi):
        if self.graph.params.state_action:
            P_pi = self.get_P_pi_sa_matrix(pi).reshape(self.n_states*self.n_actions, self.n_states*self.n_actions)
            # P_pi = self.get_P_pi_s_pis_matrix(pi).reshape(self.n_states * self.n_actions, self.n_states * self.n_actions)
        else:
            P_pi = self.compute_P_pi_ss(pi)
        graph = P_pi
        if self.graph.params.laplacian:
            graph = get_rw_laplacian(A=None, P=P_pi, normalized=self.graph.params.normalized_P)
        if self.params.features.decomp == 'SVD':
            phi_u, phi_v, _ = get_singular_rep(graph, inv=not self.graph.params.laplacian, k=self.k)
            vectors = {'L': phi_u, 'R': phi_v, 'LR': np.concatenate([phi_u, phi_v], axis=1)}
            phi = vectors[self.params.features.vector]
        else:
            phi, _ = get_eigen_rep(graph, inv=not self.graph.params.laplacian, k=self.k)
        if self.graph.params.state_action:
            phi = phi.reshape(self.n_states, self.n_actions, -1)
        return np.real(phi)