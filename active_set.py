import numpy as np
from typing import Callable
import time


def get_solution(M, y):
    x = np.linalg.solve(M[:, :M.shape[0]], y)
    x = np.pad(x, (0, len(x)))
    return x


def get_last_solution(y):
    x = np.zeros(20)
    for i, _y in enumerate(y):
        x[i * 2] = _y
        i += 2
    return x


def get_problem_M(m, n, seed):
    np.random.seed(seed)
    M = np.zeros((m, n))
    stepsize = int(n / m)
    for row in range(m):
        M[row, row * stepsize: row * stepsize + stepsize] = 1
    M += np.random.random(size=(m, n))
    eig_values = np.real(np.linalg.eig(M.T @ M)[0])
    spectral_norm = np.sqrt(np.max(eig_values))
    y = np.abs(np.random.uniform(spectral_norm, 10 + spectral_norm, m))
    G = M.T @ M
    c = -M.T @ y
    return M, y, G, c


def get_problem_M_tilde():
    M = []
    for i in range(5):
        current_row = []
        next_row = []
        for j in range(5):
            if i == j:
                current_row.append([1, 1, 0, 0])
                next_row.append([0, 0, 1, 1])
            else:
                current_row.append(np.zeros(4))
                next_row.append(np.zeros(4))
        M.append(current_row)
        M.append(next_row)

    M = np.array(M)
    M = M.reshape(10, 20)

    y = np.array([1, -2, 3, -4, 5, -5, 4, -3, 2, -1])
    G = M.T @ M
    c = -M.T @ y

    return M, y, G, c


def get_constraints(starting_points, n, G, c):
    A = np.zeros(shape=(2 * n + 1, 2 * n))
    x = starting_points[0]
    for i in range(len(x)):
        A[i, i] = 1
        A[i, i + len(x)] = 1
        A[i + len(x), i] = -1
        A[i + len(x), i + len(x)] = 1
    for i in range(len(x), len(x) * 2):
        A[2 * n, i] = -1

    b = np.zeros(2 * n + 1)
    b[2 * n] = -1

    new_G = np.pad(G, (0, n))
    new_c = np.pad(c, (0, n))
    for i, x in enumerate(starting_points):
        starting_points[i] = np.append(x, np.abs(x))

    return A, b, starting_points, new_G, new_c


def calc_alpha(
        b: np.ndarray,
        A: np.ndarray,
        x_k: np.ndarray,
        p_k: np.ndarray,
        nW_idx: list
):
    """Computes the step length for the active set method.

    :param b: Encoding of constraint equalities
    :param A: Encoding of constraints
    :param x_k: Current point
    :param p_k: Update direction
    :param nW_idx: Non-active constraints indices
    :return: found alpha, blocking constraint index
    """
    numerator = b[nW_idx] - A[nW_idx] @ x_k
    denominator = A[nW_idx] @ p_k
    ratio = numerator / denominator

    if len(ratio) > 0:
        blocking_index = nW_idx[np.argmin(ratio)]
        alpha_k = min(1, np.amin(ratio))
        return alpha_k, blocking_index
    else:
        alpha_k = 1
        return alpha_k, None


def active_set_method(
        G: np.ndarray,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        x_0: np.ndarray,
        f: Callable,
        max_iter=1e4
):
    """Finds the optimal solution to a quadratic problem.

    :param G: 'Hessian' matrix
    :param c: Vector for linear variables
    :param A: Encoding of constraints
    :param b: Encoding of constraint equalities
    :param x_0: Starting point
    :param f: Objective function
    :param max_iter: failsafe if no optimum is found
    :return: solution, f(solution), stopping criterion, number of iterations, time
    """
    time_start = time.perf_counter_ns()
    x_k = x_0.copy()
    W_k = list(np.where(A @ x_k - b == 0)[0])

    for k in range(1, int(max_iter) + 1):
        # solve 16.39 to find p_k
        A_sub = A[W_k]

        KKT_mat = np.vstack((
            np.hstack((G, A_sub.T)),
            np.hstack((A_sub, np.zeros(shape=(A_sub.shape[0], A_sub.shape[0]))))
        ))
        h = np.zeros(shape=len(W_k))
        g = c + G @ x_k

        neg_p_k_lamda_opt, _, _, _ = np.linalg.lstsq(KKT_mat, np.concatenate([g, h]), rcond=None)

        # Extracting p_k and lamda_opt
        p_k = -neg_p_k_lamda_opt[:len(x_k)]
        lambda_opt = neg_p_k_lamda_opt[len(x_k):]

        if np.allclose(p_k, np.zeros_like(p_k)):
            lambda_hat = np.zeros(A.shape[0])
            lambda_hat[W_k] = lambda_opt

            # stopping condition, else adjust active constraint set
            if all(lambda_hat[W_k] >= 0):
                time_end = time.perf_counter_ns()
                x_k = x_k[:len(x_k) // 2]
                f_k = f(x_k)
                return x_k, f_k, "Optimum found", k, time_end - time_start
            else:
                j = np.argmin(lambda_hat)
                W_k.remove(j)
        else:
            # get non active constraints
            all_idx = []
            for i in range(A.shape[0]):
                if i not in W_k:
                    all_idx.append(i)

            inactive_i = np.setdiff1d(np.arange(0, A.shape[0]), np.array(W_k))
            final_inactive_i = inactive_i[np.where(A[inactive_i] @ p_k < -1e-5)[0]]

            # get alpha with blocking constraint index
            alpha_k, blocking_constraints = calc_alpha(b, A, x_k, p_k, final_inactive_i)

            # update x
            x_k = x_k + alpha_k * p_k

            # add blocking constraints if they exist
            if alpha_k < 1:
                W_k.append(blocking_constraints)

    time_end = time.perf_counter_ns()
    f_k = f(x_k)
    return x_k, f_k, "Too many iterations", k, time_end - time_start
