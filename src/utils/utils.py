import io
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Iterable, Callable, Optional


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() == 'none':
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def int_or_none(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'")


def float_or_none(value):
    if value == 'None':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: '{value}'")


def float_range(min_val, max_val, include_min=True, include_max=False):
    def check_float(value):
        try:
            fvalue = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not a valid float")

        if include_min:
            if fvalue < min_val:
                raise argparse.ArgumentTypeError(f"Value must be >= {min_val}")
        else:
            if fvalue <= min_val:
                raise argparse.ArgumentTypeError(f"Value must be > {min_val}")

        if include_max:
            if fvalue > max_val:
                raise argparse.ArgumentTypeError(f"Value must be <= {max_val}")
        else:
            if fvalue >= max_val:
                raise argparse.ArgumentTypeError(f"Value must be < {max_val}")
        return fvalue

    return check_float


def safe_max(a, b):
    if a is None and b is None:
        return 0  # or any default value you prefer
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def is_real(matrix):
    return int(np.isreal(matrix).all())


def is_non_negative(matrix, tol=1e-10):
    return int((matrix >= -tol).all())


def is_stochastic(matrix, axis=1):
    non_negative = np.all(matrix >= 0)
    row_sums_to_one = np.allclose(np.sum(matrix, axis=axis), 1, atol=1e-8, rtol=1e-5)
    return int(non_negative and row_sums_to_one)


def is_orthogonal(matrix):
    product = np.dot(matrix.T, matrix)
    identity = np.eye(matrix.shape[1])
    return int(np.allclose(product, identity, atol=1e-8, rtol=1e-5))


def round_close(U, val=3):
    U[np.isclose(U, 0)] = 0
    return U.round(val)


def is_sim(U, V, atol=1e-5):
    assert U.shape[1] == V.shape[1]
    U, V = round_close(U), round_close(V)
    sim = np.zeros(U.shape[1], dtype=bool)
    for i in range(len(sim)):
        sim[i] = np.allclose(U[:, i], V[:, i], atol=atol) or np.allclose(U[:, i], -V[:, i], atol=atol)
    return np.all(sim)


def is_eig_sim_mat(U, V, S, atol=1e-1):
    U[np.isclose(U, 0)], V[np.isclose(V, 0)] = 0, 0
    clusters = cluster_eigenvalues(S)
    sim = []
    for cluster in clusters:
        if len(cluster) == 1:
            i = cluster[0]
            sim.append(np.allclose(U[:, i], V[:, i], atol=atol)
                               or np.allclose(U[:, i], -V[:, i], atol=atol))
        else:
            n_dims = len(cluster)
            U_clus = U[:, cluster]
            V_clus = V[:, cluster]
            sim_mat = np.zeros((n_dims, n_dims))
            for i in range(n_dims):
                for j in range(n_dims):
                    sim_mat[i, j] = (np.allclose(U_clus[:, i], V_clus[:, j], atol=atol)
                               or np.allclose(U_clus[:, i], -V_clus[:, j], atol=atol))
            if sum(sim_mat.sum(axis=1) == 1) > 0:
                sim.append(True)
            else:
                sim.append(False)
    return np.all(sim)


def cluster_eigenvalues(S, tol=1e-5):
    clusters = []
    current_cluster = [0]
    last_val = S[0]
    for i in range(1, len(S)):
        if np.abs(S[i] - last_val) < tol:
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]
            last_val = S[i]
    clusters.append(current_cluster)
    return clusters


def plt_to_image(plt):
    # Save the plot to a numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_array = np.array(image)
    buf.close()
    return image_array


def get_transition_matrix(A, normalized=False):
    d = np.sum(A, axis=1)
    if normalized:
        # Compute the normalized transition matrices (symmetric)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        P = D_inv_sqrt @ A @ D_inv_sqrt
    else:
        # Compute the unnormalized transition matrices (asymmetric)
        D_inv = np.diag(1.0 / d)
        P = D_inv @ A
    return P


def get_rw_laplacian(A=None, P=None, normalized=False):
    assert P is not None or A is not None
    if P is None:
        P = get_transition_matrix(A, normalized=normalized)
    return np.eye(P.shape[0]) - P


def get_laplacian(A):
    degree = A.sum(1)
    return np.diag(degree) - A


def get_successor_matrix(P, gamma):
    I = np.eye(P.shape[0])
    M = np.linalg.inv(I - gamma * P)
    return M


def get_det_policy(stoch_pi):
    pi = np.zeros_like(stoch_pi)
    pi[np.arange(stoch_pi.shape[0]), np.argmax(stoch_pi, axis=1)] = 1
    return pi


def get_policy_changes(pi, old_pi):
    old_pi, pi = round_close(old_pi), round_close(pi)
    return np.sum(old_pi != pi), np.sum(get_det_policy(pi) != get_det_policy(pi))


def plot_xy(iters, steps, title):
    plt.clf()
    plt.plot(iters, steps)
    plt.xlabel('Iteration')
    plt.ylabel('Total Steps')
    plt.title(title)
    plt.show()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def round_up_to_multiple(value: int, multiple: int, max_value: Optional[int] = None) -> int:
    rounded = ((value + multiple - 1) // multiple) * multiple
    if max_value is not None:
        rounded = min(rounded, max_value)
    return int(rounded)


class DotDict(dict):
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            # Recursively wrap nested dictionaries into DotDict
            value = DotDict(value)
            self[key] = value  # Cache the converted dictionary
        return value

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DotDict(value)
        self[key] = value
    