import numpy as np
import matplotlib.pyplot as plt
from MDAB_algorithm import multi_grad_DAB_step
from blahut_arimoto import blahut_arimoto
from utils import (restore_file, plot_from_fig, calc_pyk, point_kl, return_corners, return_2_corners,
                   calc_full_multinomial_channel)


def map_colors(p3dc, func, cmap='viridis', cmin=0.0, cmax=1.0):
    """
    Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """

    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from numpy import array

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = array([array((x[s], y[s], z[s])).T for s in slices])

    # compute the barycentres for each triangle
    xb, yb, zb = triangles.mean(axis=1).T

    # compute the function in the barycentres
    values = func(xb, yb, zb)

    # usual stuff
    norm = Normalize()
    # colors = get_cmap(cmap)(norm(values))

    # Get the colormap and extract a sub-range
    original_cmap = get_cmap(cmap)
    new_cmap = LinearSegmentedColormap.from_list(
        f'{cmap}_sub',
        original_cmap(np.linspace(cmin, cmax, 256))
    )

    # Apply the colormap
    colors = new_cmap(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)


def plot_simplex():
    n = 7
    new_x = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.68185518, 0.31814482, 0]), np.array([1, 0, 0])]
    max_x = [0.6165029,  0.19174855, 0.19174855]
    p_y_k, I, r = calc_pyk(new_x, n)

    def f_3d_vec(x, y, z):
        return [point_kl(n, [x[i], y[i], z[i]], p_y_k) for i in range(len(x))]

    n_x = 50
    n_y = 50
    xd = np.linspace(0, 1, n_x)
    x = 1 - np.logspace(-10, -1, 20)
    xd = np.concatenate([xd, x], 0)

    yd = np.linspace(0, 1, n_y)
    y = np.logspace(-10, -1, 20)
    yd = np.concatenate([yd, y], 0)
    x, y = np.meshgrid(xd, yd)

    x = np.ravel(x)
    y = np.ravel(y)
    xy = list(zip(x, y))
    triangle = list(filter(lambda a: a[0] + a[1] <= 1, xy))
    t_len = len(triangle)
    t_x, t_y = zip(*triangle)
    t_z = [1- t_x[i]-t_y[i] for i in range(t_len)]

    fig = plt.figure()
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.style"] = 'italic'
    plt.rcParams['font.family'] = 'serif'

    ax = plt.axes(projection='3d')
    p3dc = ax.plot_trisurf(t_x, t_y, t_z, alpha=0.5)

    # change the face colors
    mappable = map_colors(p3dc, f_3d_vec, 'BuPu')
    plt.colorbar(mappable, shrink=0.67, aspect=16.7)
    ax.view_init(azim=45, elev=20)

    alpha = 50

    # x_0 = x_1
    zline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*zline
    xline = 0.5 - 0.5*zline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_1 = x_2
    xline = np.linspace(0, 1, alpha)
    yline = 0.5 - 0.5*xline
    zline = 0.5 - 0.5*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_0 = x_2
    yline = np.linspace(0, 1,  alpha)
    xline = 0.5 - 0.5*yline
    zline = 0.5 - 0.5*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_2 = 0
    yline = np.linspace(0, 1,  alpha)
    xline = 1 - yline
    zline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_1 = 0
    xline = np.linspace(0, 1,  alpha)
    zline = 1 - xline
    yline = 0*xline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    # x_0 = 0
    yline = np.linspace(0, 1,  alpha)
    zline = 1 - yline
    xline = 0*yline
    ax.plot3D(xline, yline, zline, 'k', linewidth=2, alpha=1)

    symbols = np.transpose(new_x)
    ax.scatter(symbols[0], symbols[1], symbols[2], c='teal', s=70, marker='o', edgecolor='k', alpha=1, label='Input distribution mass point')
    ax.scatter(max_x[0], max_x[1], max_x[2], c='purple', s=70, marker='X', edgecolor='k', alpha=1, label='KL divergence maximizer')

    ax.set_xlabel("$x_{1}$", fontsize="18")
    ax.set_ylabel("$x_{2}$", fontsize="18")
    ax.set_zlabel("$x_{3}$", fontsize="18")

    plt.legend(fontsize="16", bbox_to_anchor=(0., 0.4, 0.5, 0.5))
    plt.show(block=False)


def plot_approximation_capacity_multidim(max_n, fix_x, label):
    C_vec = []
    for n in range(1, max_n + 1):
        p = calc_full_multinomial_channel(fix_x, n)
        C, r = blahut_arimoto(np.asarray(p))
        C_vec.append(C)
    print("approx C_vec: ", C_vec)
    plt.plot(range(1, max_n + 1), C_vec, color='purple', label=str(label)) # , label='resolution = ' + str(label))


def compare_capacity_4d():
    f1 = restore_file('results/multidim.pickle')
    f = plt.figure()

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    plot_from_fig(f1, sub=0, label="M-DAB")
    max_n = 10

    fix_x = return_corners(dim=4)
    plot_approximation_capacity_multidim(max_n, fix_x, label="Uniform composite")

    x_range = [a for a in range(1, max_n + 1)]
    plt.plot(x_range, [np.log2(4)]*max_n, '--k', label='Non composite bound')
    plt.plot(x_range, [np.log2(15)]*max_n, 'k',linestyle='dashdot', label='Uniform composite bound')

    plt.xlim([1, 10])
    plt.legend(fontsize="16", bbox_to_anchor=(0.30, -0.05, 0.5, 0.5)) #loc="lower right" ) # bbox_to_anchor=(0.3, 0., 0.5, 0.5))
    plt.ylabel("$C_{n,k=4}$", fontsize="18")  # "Capacity"
    plt.xlabel("$n$", fontsize="18")  # Number of Multinomial Trials
    plt.grid()
    plt.show(block=False)


def plot_law(scale_vec):

    max_input = 100 # 0
    plt.figure()

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    # Creating color map
    my_cmap = plt.get_cmap('BuPu')

    # Define the normalization for the colorbar
    vmin = scale_vec[0][0]
    vmax = scale_vec[-1][0]

    for scale_input in scale_vec:
        dim = scale_input[0]
        C_vec = scale_input[1]
        num_point_vec = scale_input[2]

        assert len(C_vec) == len(num_point_vec)
        label = '$k = '+ str(dim) +"$" # dimension
        sc = plt.plot(num_point_vec, C_vec, 's', markersize=7, mec='k', label=label, alpha=1, color=my_cmap((dim-vmin+1) / (vmax-vmin+1)))
        if num_point_vec[-1] > max_input:
            max_input = num_point_vec[-1]

    plt.plot(range(1, max_input + 1), [3/4*np.log2(a) for a in range(1, max_input + 1)], '-k', label='$ \\frac{3}{4} \log(x)$')  # , linestyle='dashdot'
    plt.plot(range(1, max_input + 1), [np.log2(a) for a in range(1, max_input + 1)], '--k', label='$\log(x)$')

    plt.xscale('log')
    plt.xlabel('$m$', fontsize="18")  # 'Support size'
    plt.ylabel('$C_{n,k}$', fontsize="18")  # Capacity
    plt.legend(fontsize="16")
    plt.grid(which="both")
    plt.xlim([1, max_input])

    plt.show()


def log_plot(C_vec, num_point_vec):
    assert len(C_vec) == len(num_point_vec)
    max_input = num_point_vec[-1]
    plt.figure()
    plt.plot(num_point_vec, C_vec, 'x', label='DAB')
    plt.plot(range(1, max_input + 1), [3/4*np.log2(a) for a in range(1, max_input + 1)], '--k', label='3/4*log(x)')
    plt.xscale('log')
    plt.xlabel('input size')
    plt.ylabel('mutual information')
    plt.title('DAB algorithm')
    plt.legend()
    plt.grid()
    plt.show(block=False)


def plot_scaling_law(scale_law_params, grad="max", threshold=1e-4, global_max_method="shgo", line_search_params=None,
                   clean_zeros=0, move_to_lines=0):

    scale_vec = []
    for i in range(len(scale_law_params)):
        param = scale_law_params[i]
        dim = param[0]
        max_n = param[1]
        print("clac multi DAB for dim = %d" % dim, " with method: " + grad)
        if line_search_params is not None:
            print("line_search: ", line_search_params[0], "constant_step: ", line_search_params[1], "min_step: ",
                  line_search_params[2], "over_shoot: ", line_search_params[3], "init: ", line_search_params[4],
                  "method: ", line_search_params[5])

        y = return_2_corners(dim=dim)
        y = [np.array(a) for a in y]
        print("input: ", y)

        iter_vec = []
        C_vec = []
        D_vec = []
        num_point_vec = []
        time_vec = []
        r_vec = []
        y_vec = []

        for n in range(1, max_n + 1):
            print(f"{n = }")
            if grad == "max" or grad == "print_kl":
                C, r, y, iter_num, D, num_point, time = multi_grad_DAB_step(y, n, init_threshold=threshold, method=grad,
                                                                            global_max_method=global_max_method,
                                                                            line_search_params=line_search_params,
                                                                            clean_zeros=clean_zeros, move_to_lines=move_to_lines)
            else:
                return -1
            iter_vec.append(iter_num)
            C_vec.append(C)
            D_vec.append(D)
            num_point_vec.append(num_point)
            time_vec.append(time)
            r_vec.append(r)
            y_vec.append(y)

        print("r_vec ", r_vec)
        print("y_vec ", y_vec)
        print("C_vec ", C_vec)
        print("num_point_vec ", num_point_vec)

        log_plot(C_vec, num_point_vec)
        scale_vec.append([dim, C_vec, num_point_vec])
    plot_law(scale_vec)
