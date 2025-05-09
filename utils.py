import pickle
import matplotlib.pyplot as plt
import itertools
import operator
import numpy as np
from scipy.special import factorial, binom
from sympy.utilities.iterables import multiset_permutations
from blahut_arimoto import blahut_arimoto

# ### Files ### #
def restore_file(file_path):
    # Restore the Figure object from the file
    with open(file_path, 'rb') as f:
        fig = pickle.load(f)

    return fig


def plot_restored_file(file_path, block=True):
    f1 = restore_file(file_path)
    plt.show(block=block)


def plot_from_fig(f, sub=1, label=None):
    ax = f.get_axes()[sub]
    line = ax.get_lines()[0]
    x, y = line.get_data()
    if label is None:
        plt.plot(x, y)
    else:
        plt.plot(x, y, label=label, color='teal')

# ### Information ### #
def kl_divergence(p, q):
    eps = 1e-16
    return np.sum(np.where(p != 0, p * np.log2(eps + p / (q + eps)), 0))


def point_kl(n, x, p_y_k):
    p = calc_multinomial_channel(np.array([x]), n)
    return kl_divergence(p, p_y_k)


def binomial_p(n, x, y):
    return binom(n, y)*(x**y)*((1-x)**(n-y))


def calc_binomial_channel(x, n):
    y = np.array(range(n+1))
    p_y_x = np.asarray([binomial_p(n, i, y) for i in x])
    return p_y_x


# r - number of balls
def combinations_with_replacement_counts(r, n):
    size = n + r - 1
    for indices in itertools.combinations(range(size), r-1):
        starts = [0] + [index+1 for index in indices]
        stops = indices + (size,)
        yield np.array(list(map(operator.sub, stops, starts)))


def multinomial_coeff(c):
    return factorial(c.sum()) / factorial(c).prod()


def multinomial_p(n, x_vec, y_vec):
    assert sum(y_vec) == n
    if np.abs(sum(x_vec)) - 1 > 1e-4 :
        x_vec = x_vec/sum(x_vec)
    assert len(x_vec) == len(y_vec)
    power_vec = x_vec ** y_vec
    ret_val = multinomial_coeff(y_vec)*(power_vec.prod())
    if ret_val < 0:
        ret_val = 0
    return ret_val


def calc_multinomial_channel(x, n):
    dim = len(x[0])
    p_y_x = []

    for symbol in x:
        p_y_symbol = [multinomial_p(n, symbol, permutation) for permutation in combinations_with_replacement_counts(dim, n)]
        p_y_symbol = p_y_symbol/sum(p_y_symbol)
        p_y_x.append(p_y_symbol)
    return p_y_x


def return_full(x):
    full = []
    for point in x:
        full.extend([np.array(p) for p in multiset_permutations(point)])
    return [np.array(a) for a in np.unique(full, axis=0)]


def calc_full_multinomial_channel(x, n):
    return calc_multinomial_channel(return_full(x), n)


def return_2_corners(dim):
    corners = []

    g = [1]
    g.extend([0] * (dim - 1))
    corners.append(np.array(g))

    g = [1 / dim] * dim
    corners.append(np.array(g))
    return np.unique(corners, axis=0)


def return_corners(dim):
    corners = []
    for d in range(1, dim + 1):
        g = [1 / d] * d
        g.extend([0] * (dim - d))
        corners.append(np.array(g))
    return np.unique(corners, axis=0)


def calc_pyk(x, n):
    p = calc_full_multinomial_channel(x, n)
    I, r = blahut_arimoto(np.asarray(p))
    p_y_k = np.matmul(np.array([r]), p)
    return p_y_k, I, r
