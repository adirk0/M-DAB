import time
import numpy as np
from utils import kl_divergence, calc_pyk, return_corners, calc_full_multinomial_channel, calc_multinomial_channel
from blahut_arimoto import blahut_arimoto
from scipy.optimize import minimize, Bounds, shgo, LinearConstraint


def find_beneath_kl(x, max_x):

    dim = len(x[0])
    middle = np.array([1 / dim] * dim)
    c = kl_divergence(max_x, middle)

    guess_vec = [a for a in x]
    guess_vec = np.concatenate((guess_vec, return_corners(dim)), 0)
    guess_vec = np.unique(guess_vec, axis=0)

    closest_value = None
    closest_dist = float('inf')

    for value in guess_vec:
        result = kl_divergence(value, max_x)
        if result < closest_dist and kl_divergence(value, middle) <= kl_divergence(max_x, middle):
            closest_value = value
            closest_dist = result

    return closest_value


def add_delta_grad(delta, new_x, grad):
    assert len(new_x) == len(grad)
    x = new_x + delta * grad
    x = [a/sum(a) for a in x]

    assert len(x) == len(new_x)
    return x


def calc_new_I(delta, new_x, grad, n):
    x = add_delta_grad(delta, new_x, grad)
    p = calc_full_multinomial_channel(x, n)
    I, r = blahut_arimoto(np.asarray(p))
    return I


# move x[idx] in the direction to dest
def add_line_search(x, idx, dest, n, line_search_params=None):
    if line_search_params is None:
        line_search_params = [1, 0, 0, 1, 1e-3, "shgo"]

    line_search = line_search_params[0]
    constant_step = line_search_params[1]
    min_step = line_search_params[2]
    over_shoot = line_search_params[3]
    init = line_search_params[4]
    method = line_search_params[5]

    assert len(x[0]) == len(dest)
    amount = len(x)
    dim = len(x[0])
    if (dest == x[idx]).all():
        return x, 0
    direction = dest - x[idx]
    grad = [np.array([0] * dim)] * amount
    grad[idx] = direction

    if line_search:

        def cost_fun(d):
            return -calc_new_I(d, x, grad, n)
        bounds = Bounds([min_step], [1])
        if method == "shgo":
            res = shgo(cost_fun, bounds)
        elif method == "bfgs":
            res = minimize(cost_fun, init, bounds=bounds)
        else:
            print("not such method for line search")
            assert False

        delta = over_shoot * res.x

    else:
        delta = constant_step

    return add_delta_grad(delta, x, grad), delta


# update only forward
def update_only_forward(x, max_x, n):
    assert len(x[0]) == len(max_x)

    delta = 0
    add = 0

    closest_value = find_beneath_kl(x, max_x)
    is_in_list = np.any(np.all(closest_value == x, axis=1))
    if is_in_list: # adjust x
        nearest_max_idx = np.where(np.all(closest_value == x, axis=1))[0][0]
        new_x, delta = add_line_search(x, nearest_max_idx, max_x, n, line_search=0, constant_step=1e-3)

    else:  # add point
        new_x = np.concatenate((x, np.array([closest_value])), 0)
        new_x = [np.array(a) for a in new_x]
        add = 1

    return new_x, delta, add


# update nearest - looking for corner to add
def update_x(x, max_x, n, line_search_params=None):
    assert len(x[0]) == len(max_x)
    dim = len(x[0])

    delta = 0
    add = 0

    dist_vec = [kl_divergence(a, max_x) for a in x]
    nearest_max_idx = np.array(dist_vec).argmin()
    nearest_max_dist = dist_vec[nearest_max_idx]

    corners = return_corners(dim)
    dist_vec = [kl_divergence(a, max_x) for a in corners]
    nearest_corner_idx = np.array(dist_vec).argmin()
    nearest_corner_dist = dist_vec[nearest_corner_idx]

    if nearest_max_dist <= nearest_corner_dist:  # adjust x
        new_x, delta = add_line_search(x, nearest_max_idx, max_x, n, line_search_params)

    else:  # add point
        new_x = np.concatenate((x, np.array([corners[nearest_corner_idx]])), 0)
        new_x = [np.array(a) for a in new_x]
        add = 1

    return new_x, delta, add


def join_near_edges(x, th=1e-3):
    for i in range(len(x)-1):
        if abs(x[i]-x[i+1]) < th:
            x[i + 1] = x[i]
    return x


def nullify(x, th=1e-3):
    return [a if a >= th else 0 for a in x]


def find_global_shgo(dim, n, p_y_k):
    def cost_fun(x):
        p = calc_multinomial_channel(np.array([x]), n)
        return -kl_divergence(p, p_y_k)

    bounds = Bounds([0] * dim, [1] * dim)

    linear_constraint1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    def linear_constraint2(x):
        return np.dot(np.eye(dim - 1, dim) - np.eye(dim - 1, dim, 1), x) - np.ones(dim - 1)

    linear_constraints = [{'type': 'ineq', 'fun': linear_constraint2}]

    res = shgo(cost_fun, bounds, constraints=[linear_constraint1] + linear_constraints)

    return -res.fun, res.x


def find_kl_max_init(dim, n, p_y_k, x0):
    def cost_fun(x):
        p = calc_multinomial_channel(np.array([x]), n)
        return -kl_divergence(p, p_y_k)

    bounds = Bounds([0]*dim, [1]*dim)
    linear_constraint1 = LinearConstraint([1]*dim, [1], [1])
    linear_constraint2 = LinearConstraint(np.eye(dim - 1, dim) - np.eye(dim - 1, dim, 1), [0]*(dim-1), [1]*(dim-1))
    res = minimize(cost_fun, x0, constraints=[linear_constraint1, linear_constraint2], bounds=bounds)

    return -res.fun, res.x


def search_global_by_hand(dim, n, p_y_k, x, threshold):
    D = 0
    guess_vec = [a for a in x]
    guess_vec = np.concatenate((guess_vec, return_corners(dim)), 0)
    guess_vec = np.unique(guess_vec, axis=0)

    # search for different global maximum:
    for i in range(len(guess_vec) - 1):
        for j in range(i, len(guess_vec)):  # use i+1 if already handle corners
            for alpha in [0.5]:
                x0 = guess_vec[i] * alpha + guess_vec[j] * (1 - alpha)
                try_D, try_x = find_kl_max_init(dim, n, p_y_k, x0)
                if try_D >= D + threshold:
                    D = try_D
                    max_x = try_x
    return D, max_x


def multi_grad_inner_step(dim, n, p_y_k, x, threshold, method='max', global_max_method="shgo", line_search_params=None,
                          clean_zeros=0, move_to_lines=0):

    if global_max_method == "shgo":
        D, max_x = find_global_shgo(dim, n, p_y_k)
    elif global_max_method == "hand":
        D, max_x = search_global_by_hand(dim, n, p_y_k, x, threshold)
    else:
        print("not such global max method")
        assert False

    if method == 'forward':
        x, delta, add = update_only_forward(x, max_x, n)  # update_x(x, max_x, n)
    elif method == 'max':
        x, delta, add = update_x(x, max_x, n, line_search_params)
    else:
        print("not such update method")
        assert False

    if clean_zeros:
        x = [nullify(a, clean_zeros) for a in x]

    if move_to_lines:
        x = [join_near_edges(a, move_to_lines) for a in x]

    return D, max_x, x, delta, add


def multi_grad_DAB_step(new_x, n, init_threshold=1e-3, method='max', global_max_method="shgo", line_search_params=None,
                        clean_zeros=0, move_to_lines=0):

    iter_num = 0
    max_iter = 200
    threshold = init_threshold

    t_start = time.time()

    dim = len(new_x[0])

    p_y_k, I, r = calc_pyk(new_x, n)

    # optimize the allocation of probability to the current mass point locations
    while True:
        if iter_num == 200:
            print("warning")
        assert iter_num < max_iter

        x = new_x

        # find maximum
        D, max_x, new_x, inner_delta, add = multi_grad_inner_step(dim, n, p_y_k, x, threshold, method=method,
                                                                  global_max_method=global_max_method,
                                                                  line_search_params=line_search_params,
                                                                  clean_zeros=clean_zeros, move_to_lines=move_to_lines)
        assert I < D + 1e-1

        p_y_k, I, r = calc_pyk(new_x, n)
        if D - I <= threshold:
            num_point = sum([1 for a in r if a > 1e-10])
            t_end = time.time()
            total_time = t_end - t_start
            return I, r, new_x, iter_num, D, num_point, total_time

        iter_num += 1
