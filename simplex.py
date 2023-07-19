import numpy as np
import time
from scipy.linalg import lu


def simplex_start(
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x0=None
):
    """Finds feasible point for starting the simplex method.

    :param A: encoding of constraints
    :param b: used for matrix E
    :param c: encoding of the objective
    :param x0: starting point
    :return: solution, B_idx, N_idx, f(solution), stopping criterion, number of iterations, time
    """
    m, n = A.shape
    e = np.ones((m,))
    E = np.zeros((m, m))

    # Phase I
    # E_jj = +1 if bj ≥ 0, E_jj = −1 if bj < 0
    for j in range(m):  # b is of shape m,
        E[j, j] = 1 if b[j] >= 0 else -1

    # this way a first feasible point is trivially given by
    z = np.abs(b)

    # we need to re-write the phase I problem in order to apply the simplex method further on
    p1_A = np.concatenate([A, E], -1)
    p1_c = np.concatenate([np.zeros((n,)), e], 0)
    if x0 is None:
        x0 = np.zeros((n,))

    p1_x = np.concatenate([x0, z], 0)

    # by construction the first n entries of x are 0 (active constraints w.r.t. x ≥ 0)
    p1_B_ = np.arange(n, n + m)
    p1_N_ = np.arange(n)

    # solve the phase I problem to get an initial feasible solution for phase II
    p1_x, p1_B_, p1_N_, _, _, i, duration1 = revised_simplex(p1_c, p1_A, p1_B_, p1_N_, b, p1_x)

    # if e'z is positive at this solution the original problem is infeasible
    if not np.isclose(np.sum(p1_x[n:]), 0):
        raise RuntimeError(f'Problem is infeasible ({np.sum(p1_x[n:])})')

    # Phase II
    if np.all(p1_B_ < n):
        # if all indices of the inactive set belong to some x of the original problem
        # we have the case that no artificial elements of z are remaining in the basis
        p2_x = p1_x[:n]
        p2_B_ = p1_B_

        # take those indices i ∈ {1,2,...n} not already in the inactive set
        p2_N_ = np.array(list({i for i in range(n)} - set(p1_B_)))

        sol, _, _, f_x, stop_crit, j, duration2 = revised_simplex(c, A, p2_B_, p2_N_, b, p2_x)

        return sol, f_x, stop_crit, np.array([i, j]), np.array([duration1, duration2])


def revised_simplex(
        c: np.ndarray,
        A: np.ndarray,
        B_idx: np.ndarray,
        N_idx: np.ndarray,
        b: np.ndarray,
        x=None,
        max_iter=1e6
):
    """Finds the optimal solution for a linear problem.

    :param c: encoding of the objective
    :param A: encoding of the constraints
    :param B_idx: indices of the slack variables
    :param N_idx: indices of the non-slack variables
    :param b: used if x=None
    :param x: starting point
    :param max_iter: failsafe if no solution found
    :return: solution, B_idx, N_idx, f(solution), stopping criterion, number of iterations, time
    """
    time_start = time.perf_counter_ns()
    solution = np.zeros_like(c).astype(float)

    for i in range(1, int(max_iter) + 1):
        B = A[:, B_idx]
        N = A[:, N_idx]

        if i == 1:
            if x is None:
                x_B = np.linalg.solve(B, b)
            else:
                x_B = x[B_idx]

        c_B = c[B_idx]
        c_N = c[N_idx]

        lambda_ = np.linalg.solve(B.T, c_B)

        s_N = c_N - N.T @ lambda_
        if all(s_N >= 0):
            time_end = time.perf_counter_ns()
            solution[B_idx] = x_B
            f_x = np.sum(c * solution)
            return solution, B_idx, N_idx, f_x, "Optimal point found", i, time_end - time_start

        q = np.argmin(s_N)
        q = N_idx[q]

        d = np.linalg.solve(B, A[:, q])

        if all(d <= 0):
            time_end = time.perf_counter_ns()
            return np.array([]), B_idx, N_idx, -1, "Problem is unbounded", i, time_end - time_start

        d_idx = np.where(d > 0)[0]

        p = np.argmin(x_B[d_idx] / d[d_idx])
        p = d_idx[p]
        x_q = np.min(x_B[d_idx] / d[d_idx])

        x_B = x_B - d * x_q

        solution[B_idx] = x_B
        solution[q] = x_q

        B_idx_tmp = B_idx[p]
        B_idx[B_idx == B_idx_tmp] = q
        N_idx[N_idx == q] = B_idx_tmp

        x_B[p] = x_q

    time_end = time.perf_counter_ns()
    f_x = np.sum(c * solution)
    return solution, B_idx, N_idx, f_x, "Too many iterations", i, time_end - time_start


if __name__ == "__main__":
    c = np.array([-4, -1, 0, 0])
    A = np.array([[1, 1, 1, 0],
                  [2, 0.5, 0, 1]])
    B_idx = np.array([2, 3])
    N_idx = np.array([0, 1])
    b = np.array([5, 8])
    x = np.array([1, 0, 5, 8])
    non_slack = np.array([0, 1])

    A_1 = np.array([[-13., -30., -12., 40., -42., 93., -61., -13., 74., -12., 1.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-19., 65., -75., -23., -28., -91., 48., 15., 97., -21., 0.,
                     1., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [75., 92., -18., -1., 77., -71., 47., 47., 42., 67., 0.,
                     0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [-68., 93., -91., 85., 27., -68., -69., 51., 63., 14., 0.,
                     0., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [83., -72., -66., 28., 28., 64., -47., 33., -62., -83., 0.,
                     0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [-21., 32., 5., -58., 86., -69., 20., -99., -35., 69., 0.,
                     0., 0., 0., 0., 1., 0., 0., 0., 0.],
                    [-43., -65., 2., 19., -89., 74., -18., -9., 28., 42., 0.,
                     0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [-1., -47., 40., 21., 70., -16., -32., -94., 96., -53., 0.,
                     0., 0., 0., 0., 0., 0., 1., 0., 0.],
                    [27., 31., 0., 80., -22., 43., 48., 86., -77., 41., 0.,
                     0., 0., 0., 0., 0., 0., 0., 1., 0.],
                    [17., -15., -52., -51., -31., 69., 63., 92., -5., 97., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    b_1 = np.array([94, 0, 50, 36, 34, 48, 93, 3, 98, 42])
    c_1 = np.array([72., -53., 17., 92., -33., 95., 3., -91., -79., -64., 0.,
                    0., 0., 0., 0., 0., 0., 0., 0., 0.])
    x_not_feas_1 = np.array([77, 21, 73, 0, 10, 43, 58, 23, 59, 2, 62, 35, 94, 67, 82, 46, 20,
                             81, 50, 27])
    x_feas_1 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 94., 0., 50.,
                         36., 34., 48., 93., 3., 98., 42.])

    B_ = np.array(range(10, 20))
    N_ = np.array(range(10))
    print(simplex_start(A, b, c, x))
    non_slack = np.array(range(10))

    print(revised_simplex(c_1, A_1, B_, N_, b_1, x=x_not_feas_1))
    print(simplex_start(A_1, b_1, c_1, x_feas_1))
    print(simplex_start(A_1, b_1, c_1, x_not_feas_1))
    print(revised_simplex(c, A, B_idx, N_idx, b, x=x))
