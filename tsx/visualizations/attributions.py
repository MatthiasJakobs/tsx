import matplotlib.pyplot as plt 
from matplotlib import cm
import numpy as np

from tsx.utils import to_numpy
from tsx.visualizations.utils import calc_optimal_grid

def plot_cam(data, attribution):
    data = np.squeeze(to_numpy(data))
    attribution = np.squeeze(to_numpy(attribution))
    if len(data.shape) == 1:
        nr_images = 1
    else:
        nr_images = data.shape[0]    

    xs = np.arange(data.shape[-1])
    colors = cm.get_cmap('viridis')
    rows, cols = calc_optimal_grid(nr_images)

    plt.figure()
    for i in range(nr_images):
        plt.subplot(rows, cols, i+1)

        if nr_images == 1:
            ittr = zip(data, attribution, xs)
            plt.plot(data, color="black", zorder=1)
        else:
            ittr = zip(data[i], attribution[i], xs)
            plt.plot(data[i], color="black", zorder=1)

        for y_i, c_i, x_i in ittr:
            plt.scatter(x_i, y_i, color=colors(c_i), zorder=2, picker=True)

        plt.colorbar()


    plt.show()




