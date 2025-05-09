from utils import plot_restored_file
from visualize import plot_simplex, compare_capacity_4d, plot_scaling_law
import matplotlib.pyplot as plt

def generate_all_figures():
    """ distance measure """
    plot_simplex()

    """ composite vs uniform """
    compare_capacity_4d()

    """ scaling law """
    # line_search_params = [line_search, constant_step, min_step, over_shoot, init, method]
    line_search_params =   [          1,             0,     5e-2,          1,  1e-1, "shgo"]
    params =  [[2, 35], [3, 20], [4, 10], [5, 10]]
    plot_scaling_law(params, threshold=1e-4, global_max_method="hand", line_search_params=line_search_params,
                     clean_zeros=1e-3, move_to_lines=1e-3)
    plt.show()


def load_figures_data():
    """ distance measure """
    plot_restored_file('results/simplex.pickle')

    """ composite vs uniform """
    plot_restored_file('results/compare_uniform.pickle')

    """ scaling law """
    plot_restored_file('results/scaling_law.pickle')


if __name__ == "__main__":
    load_figures_data()