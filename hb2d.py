from criteria import hoek_brown
import numpy as np

from norm import norm_sigma
from support import distance, angle

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    min_x = -1000
    max_x = 1000
    n = 1001
    hb_xs = np.linspace(min_x, max_x, n)
    hb_ys = [hoek_brown(x, x) for x in hb_xs]

    plt.plot(hb_xs, hb_ys)

    sigma = [-500, -500, -700]
    a = angle([1, 1, 0], [1, 0, 0])
    sigma_max_1 = hoek_brown(sigma[1], sigma[2])
    sigma_max = [sigma_max_1, sigma[1], sigma[2]]
    d = distance(sigma, sigma_max)
    sigma_n = norm_sigma(sigma)
    plt.scatter(sigma[2], sigma[0], c='red')
    plt.scatter(sigma_max[2], sigma_max[0], c='green')
    plt.scatter(sigma_n[2], sigma_n[0], c='blue')
    plt.plot([0, 0], [min_x, max_x], color='black', alpha=1, linewidth=0.5)
    plt.plot([min_x, max_x], [0, 0], color='black', alpha=1, linewidth=0.5)
    plt.xlim(min_x*1.05, max_x*1.05)
    plt.ylim(min_x*1.05, max_x*1.05)
    plt.show()
