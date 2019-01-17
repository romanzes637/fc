import math

from matplotlib import cm

from criteria import hoek_brown


def hoek_brown_min_point(mi=20, d=0.5, gsi=85, sigma_ci=200):
    mb = mi * math.exp((gsi - 100) / (28 - 14 * d))
    s = math.exp((gsi - 100) / (9 - 3 * d))
    # a = 0.5 + (math.exp(-gsi / 15) - math.exp(-20 / 3)) / 6
    sigma_3_min = -s * sigma_ci / mb
    sigma_2_min = sigma_3_min
    sigma_1_min = hoek_brown(sigma_2_min, sigma_3_min, mi=20, d=0.5, gsi=85,
                             sigma_ci=200)
    if sigma_1_min >= sigma_2_min >= sigma_3_min:
        sigma_1_min = max(sigma_2_min, sigma_3_min)
    return sigma_1_min, sigma_2_min, sigma_3_min


def plot_axes(xs, ys):
    plt.plot(xs, [0, 0], color='black', alpha=1, linewidth=0.5)
    plt.plot([0, 0], ys, color='black', alpha=1, linewidth=0.5)
    plt.grid()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    min_c = -500
    max_c = 500
    n = 10001
    nc = 5
    min_mi = 4  # 4
    max_mi = 33  # 33
    min_sigma_ci = 1  # 0
    max_sigma_ci = 250  # 250
    min_d = 0  # 0
    max_d = 1  # 1
    min_gsi = 5  # 5
    max_gsi = 100  # 100
    mis = np.linspace(min_mi, max_mi, nc)
    sigma_cis = np.linspace(min_sigma_ci, max_sigma_ci, nc)
    ds = np.linspace(min_d, max_d, nc)
    gsis = np.linspace(min_gsi, max_gsi, nc)
    # X, Y
    xs = [min_c, max_c]
    ys = [min_c, max_c]
    cmap = cm.get_cmap('Greys')
    min_color = 0.3
    hb_xs = np.linspace(min_c, max_c, n)
    for i, mi in enumerate(mis):
        hb_ys = [hoek_brown(x, x, mi=mi) for x in hb_xs]
        plt.plot(hb_xs, hb_ys,
                 color=cmap(min_color + (1 - min_color) * (i / nc)), label=mi)
        plt.xlim(min_c * 1.05, max_c * 1.05)
        plt.ylim(min_c * 1.05, max_c * 1.05)
    plt.legend(title='mi')
    plot_axes(xs, ys)
    plt.show()
    plt.cla()
    for i, sigma_ci in enumerate(sigma_cis):
        hb_ys = [hoek_brown(x, x, sigma_ci=sigma_ci) for x in hb_xs]
        plt.plot(hb_xs, hb_ys,
                 color=cmap(min_color + (1 - min_color) * (i / nc)),
                 label=sigma_ci)
        plt.xlim(min_c * 1.05, max_c * 1.05)
        plt.ylim(min_c * 1.05, max_c * 1.05)
    plt.legend(title='sigma_ci')
    plot_axes(xs, ys)
    plt.show()
    plt.cla()
    for i, d in enumerate(ds):
        hb_ys = [hoek_brown(x, x, d=d) for x in hb_xs]
        plt.plot(hb_xs, hb_ys,
                 color=cmap(min_color + (1 - min_color) * (i / nc)), label=d)
        plt.xlim(min_c * 1.05, max_c * 1.05)
        plt.ylim(min_c * 1.05, max_c * 1.05)
    plt.legend(title='D')
    plot_axes(xs, ys)
    plt.show()
    plt.cla()
    for i, gsi in enumerate(gsis):
        hb_ys = [hoek_brown(x, x, gsi=gsi) for x in hb_xs]
        plt.plot(hb_xs, hb_ys,
                 color=cmap(min_color + (1 - min_color) * (i / nc)), label=gsi)
        plt.xlim(min_c * 1.05, max_c * 1.05)
        plt.ylim(min_c * 1.05, max_c * 1.05)
    plt.legend(title='GSI')
    plot_axes(xs, ys)
    plt.show()
    plt.cla()
