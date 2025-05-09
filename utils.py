import pickle
import matplotlib.pyplot as plt


def restore_file(file_path):
    # Restore the Figure object from the file
    with open(file_path, 'rb') as f:
        fig = pickle.load(f)

    return fig


def plot_restored_file(file_path, block=True):
    f1 = restore_file(file_path)
    plt.show(block=block)
