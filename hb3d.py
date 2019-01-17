# import math
# import numpy as np
# from matplotlib import cm
# import matplotlib.pyplot as plt
#
# cmap = cm.get_cmap('rainbow')
# max_mi = 50
# mi = 20
# D = 0.5
# gsis = range(0, 100, 20)
# mis = range(0, max_mi, 10)
# max_sigma_min = 100
# sigma_min = list(range(0, max_sigma_min, 1))
# sigma_ci = 200
# for mi in mis:
#     for gsi in gsis:
#         mb = mi * math.exp((gsi - 100) / (28 - 14 * D))
#         s = math.exp((gsi - 100) / (9 - 3 * D))
#         a = 0.5 + (math.exp(-gsi / 15) - math.exp(-20 / 3)) / 6
#         sigma_max = list(map(lambda x: x + sigma_ci * (mb * x / sigma_ci + s) ** a, sigma_min))
#         plt.plot(sigma_min, sigma_max, color=cmap(mi / max_mi))
# plt.plot(sigma_min, sigma_min, color='black')
# plt.xlim([0, max_sigma_min])
# plt.ylim(ymin=0)
# plt.show()
import math
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from criteria import hoek_brown

fig = plt.figure()
ax = fig.gca(projection='3d')


def cut_plot(minor, major, inter, color='black', alpha=1.0):
    p_minor = list()
    p_major = list()
    p_inter = list()
    for i in range(len(major)):
        if major[i] >= inter[i] >= minor[i]:
            p_minor.append(minor[i])
            p_major.append(major[i])
            p_inter.append(inter[i])
        elif len(p_minor) > 0:
            ax.plot(p_minor, p_major, p_inter, color=color, alpha=alpha)
            p_minor = list()
            p_major = list()
            p_inter = list()
    if len(p_minor) > 0:
        ax.plot(p_minor, p_major, p_inter, color=color, alpha=alpha)


def triangle(min_c, max_c, n):
    delta = max_c - min_c
    delta_n = delta / n
    delta_n_curve = delta / n_curve
    cs = np.arange(min_c, max_c + delta_n, delta_n)

    # In
    # xs = cs
    # ys = cs
    # zs = cs
    # pxs = list()
    # pys = list()
    # pzs = list()
    # for i in range(len(cs)):
    #     for x in xs:
    #         for y in ys:
    #             for z in zs:
    #                 if x >= y >= z:
    #                     pxs.append(x)
    #                     pys.append(y)
    #                     pzs.append(z)
    # ax.scatter(pzs, pxs, pys, color='grey', alpha=0.1, marker='.')

    # Edges
    x = cs_curve
    z = cs_curve
    y = cs_curve
    cut_plot(x, y, z, color='grey', alpha=0.5)

    x = [min_c] * len(cs_curve)
    z = cs_curve
    y = cs_curve
    cut_plot(x, y, z, color='grey', alpha=0.5)

    x = [min_c] * len(cs_curve)
    z = [min_c] * len(cs_curve)
    y = cs_curve
    cut_plot(x, y, z, color='grey', alpha=0.5)

    x = cs_curve
    z = [max_c] * len(cs_curve)
    y = [max_c] * len(cs_curve)
    cut_plot(x, y, z, color='grey', alpha=0.5)

    x = cs_curve
    z = [min_c] * len(cs_curve)
    y = [max_c] * len(cs_curve)
    cut_plot(x, y, z, color='grey', alpha=0.5)

    x = [min_c] * len(cs_curve)
    z = cs_curve
    y = [max_c] * len(cs_curve)
    cut_plot(x, y, z, color='grey', alpha=0.5)

    x = cs_curve
    z = cs_curve
    y = [max_c] * len(cs_curve)
    cut_plot(x, y, z, color='grey', alpha=0.5)


min_c = -1000
max_c = 1000
n = 4
n_curve = 2000
delta_n_curve = (max_c - min_c) / n_curve
cs_curve = list(np.arange(min_c, max_c + delta_n_curve, delta_n_curve))
triangle(min_c, max_c, n)


# X, Z = np.meshgrid(cs_curve, cs_curve)
# Y = np.sqrt(X ** 2) + np.sqrt(Z ** 2)
# Y = np.sqrt(X ** 2 + Z ** 2)
# a = 40
# b = 1
# c = 0.5
# d = -10
# e = 1
# f = 0.5
# g = 100
# h = 0
# j = 0
# Y = a * (b * X + h) ** c + d * (e * Z + j) ** f + g
# surf = ax.plot_surface(X, Y, Z, color='grey', linewidth=0, antialiased=False, alpha=0.1)


def criterion_surface(sigma_2, sigma_3):
    a1 = 30
    a2 = 1
    a3 = 10
    a4 = 0.5
    b1 = -1
    b2 = 1
    b3 = 0
    b4 = 0.5
    c1 = -10
    if a2 * sigma_3 + a3 >= 0 and b2 * (sigma_2 - sigma_3) + b3 >= 0:
        sigma_1 = a1 * (a2 * sigma_3 + a3) ** a4 + b1 * (b2 * (sigma_2 - sigma_3) + b3) ** b4 + c1
    else:
        sigma_1 = max(sigma_2, sigma_3)
    if sigma_1 >= sigma_2 >= sigma_3:
        return sigma_1
    else:
        return max(sigma_2, sigma_3)


def major_failure(criterion, point, color='blue', alpha=1.0):
    major_max = criterion(point[1], point[2])
    major_min = max(point[1], point[2])
    ax.scatter(point[2], major_max, point[1], color='red', marker='1')
    ax.scatter(point[2], major_min, point[1], color='green', marker='2')
    majors = [major_min, major_max, point[0]]
    majors.sort()
    minors = [point[2], point[2], point[2]]
    inters = [point[1], point[1], point[1]]
    ax.plot(minors, majors, inters, color=color, alpha=alpha)


# Hoek Brown
minor = cs_curve
inter = cs_curve
major = list(map(hoek_brown, inter, minor))
cut_plot(minor, major, inter, color='black')
for i in range(100):
    minor = cs_curve
    inter = list(map(lambda ci: ci + 15 * i, cs_curve))
    major = list(map(hoek_brown, inter, minor))
    cut_plot(minor, major, inter, color='grey', alpha=0.2)


# Compression
# minor = cs_curve
# inter = cs_curve
# major = list(map(criterion_surface, inter, minor))
# cut_plot(minor, major, inter, color='purple')
# for i in range(100):
#     minor = cs_curve
#     inter = list(map(lambda ci: ci + 10 * i, cs_curve))
#     major = list(map(criterion_surface, inter, minor))
#     cut_plot(minor, major, inter, color='purple', alpha=0.2)


# Points
def points(min_c, max_c, n):
    majors = list()
    inters = list()
    minors = list()
    for i in range(n):
        major = -1
        minor = 1
        inter = 0
        while not major >= inter >= minor:
            major = random.uniform(min_c, max_c)
            inter = random.uniform(min_c, max_c)
            minor = random.uniform(min_c, max_c)
        majors.append(major)
        inters.append(inter)
        minors.append(minor)
    return majors, inters, minors


def delta(p0, p1):
    d = list(map(lambda x, y: y - x, p0, p1))
    return d


def norm(v):
    n = math.sqrt(sum(map(lambda x: x * x, v)))
    return n


def fd(f, p, inter, minor):
    major = f(inter, minor)
    p1 = [major, inter, minor]
    r = delta(p, p1)
    d = norm(r)
    return d


def grad8_fd(f, p, inter, minor, dx):
    dx2 = dx / math.sqrt(2)
    f0 = fd(f, p, inter, minor)
    f1_pos = fd(f, p, inter + dx2, minor + dx2)
    f1_neg = fd(f, p, inter - dx2, minor - dx2)
    f1_inter_pos = fd(f, p, inter + dx, minor)
    if inter - dx >= minor:
        f1_inter_neg = fd(f, p, inter - dx, minor)
    else:
        f1_inter_neg = f0
    if inter >= minor + dx:
        f1_minor_pos = fd(f, p, inter, minor + dx)
    else:
        f1_minor_pos = f0
    f1_minor_neg = fd(f, p, inter, minor - dx)
    f1_inter_pos_minor_neg = fd(f, p, inter + dx2, minor - dx2)
    if inter - dx2 >= minor + dx2:
        f1_inter_neg_minor_pos = fd(f, p, inter - dx2, minor + dx2)
    else:
        f1_inter_neg_minor_pos = f0
    df_pos = (f1_pos - f0) / dx
    df_neg = (f1_neg - f0) / dx
    df_inter_pos_minor_neg = (f1_inter_pos_minor_neg - f0) / dx
    df_inter_neg_minor_pos = (f1_inter_neg_minor_pos - f0) / dx
    df_inter_pos = (f1_inter_pos - f0) / dx
    df_inter_neg = (f1_inter_neg - f0) / dx
    df_minor_pos = (f1_minor_pos - f0) / dx
    df_minor_neg = (f1_minor_neg - f0) / dx
    return [df_pos, df_neg, df_inter_pos_minor_neg, df_inter_neg_minor_pos, df_inter_pos, df_inter_neg, df_minor_pos, df_minor_neg]


def grad4_fd(f, p, inter, minor, dx):
    f0 = fd(f, p, inter, minor)
    f1_inter_pos = fd(f, p, inter + dx, minor)
    if inter - dx >= minor:
        f1_inter_neg = fd(f, p, inter - dx, minor)
    else:
        f1_inter_neg = f0
    if inter >= minor + dx:
        f1_minor_pos = fd(f, p, inter, minor + dx)
    else:
        f1_minor_pos = f0
    f1_minor_neg = fd(f, p, inter, minor - dx)
    df_inter_pos = (f1_inter_pos - f0) / dx
    df_inter_neg = (f1_inter_neg - f0) / dx
    df_minor_pos = (f1_minor_pos - f0) / dx
    df_minor_neg = (f1_minor_neg - f0) / dx
    return [df_inter_pos, df_inter_neg, df_minor_pos, df_minor_neg]


inter4_dx = [1, -1, 0, 0]
minor4_dx = [0, 0, 1, -1]

inter8_dx = [1 / math.sqrt(2), -1 / math.sqrt(2), 1 / math.sqrt(2), -1 / math.sqrt(2), 1, -1, 0, 0]
minor8_dx = [1 / math.sqrt(2), -1 / math.sqrt(2), -1 / math.sqrt(2), 1 / math.sqrt(2), 0, 0, 1, -1]

cmap = cm.get_cmap('Set1')


def limit_point(p0, p1, atol=0.1, maxit=10000, reverse=False):
    limit = p0
    if reverse:
        r = delta(p1, p0)
    else:
        r = delta(p0, p1)
    n = norm(r)
    nr = list(map(lambda x: x / n, r))
    if nr[0] >= nr[1] >= nr[2]:
        return None
    else:
        dl = min([limit[0] - limit[1], limit[1] - limit[2]])
        cnt = 0
        while dl > atol and cnt < maxit:
            cnt += 1
            limit[0] += nr[0]
            limit[1] += nr[1]
            limit[2] += nr[2]
            dl = min([limit[0] - limit[1], limit[1] - limit[2]])
        limit[0] -= nr[0]
        limit[1] -= nr[1]
        limit[2] -= nr[2]
    return limit


def distance(f, p, atol=0.01, maxit=1000):
    inter = p[1]
    minor = p[2]
    major = f(inter, minor)
    if major <= inter:
        major = 0
        inter = 0
        minor = 0
    pd = fd(f, p, inter, minor)  # previous step distance
    dxs = [10 * atol, 100 * atol, pd, pd / 10, pd / 100]
    ds = list()
    ps = list()
    for i, dx in enumerate(dxs):
        inter = p[1]
        minor = p[2]
        major = f(inter, minor)
        if major <= inter:
            major = 0
            inter = 0
            minor = 0
        pd = fd(f, p, inter, minor)
        dd = 2 * atol
        cnt = 0
        while abs(dd) > atol and cnt < maxit:
            cnt += 1
            # Path choosing
            grad = grad8_fd(f, p, inter, minor, dx)
            path_is_chosen = False
            while not path_is_chosen:
                if min(grad) >= 0:  # no negative grad
                    break
                min_grad_i = grad.index(min(grad))
                new_inter = inter + inter8_dx[min_grad_i] * dx
                new_minor = minor + minor8_dx[min_grad_i] * dx
                if new_inter < new_minor:  # bad stress order
                    grad[min_grad_i] = 0  # change this grad to 0 (non negative)
                else:  # choose this path
                    new_major = f(new_inter, new_minor)
                    if new_major <= new_inter:
                        grad[min_grad_i] = 0  # change this grad to 0 (non negative)
                    else:
                        inter = new_inter
                        minor = new_minor
                        path_is_chosen = True
            # Go
            major = f(inter, minor)
            d = fd(f, p, inter, minor)
            dd = d - pd
            pd = d
            # ax.scatter(minor, major, inter, color=cmap(i), marker='.')
        majors = [p[0], major]
        minors = [p[2], minor]
        inters = [p[1], inter]
        ds.append(pd)
        ps.append([major, inter, minor])
        # ax.plot(minors, majors, inters, color=cmap(i), label=i)
    i = ds.index(min(ds))
    majors = [p[0], ps[i][0]]
    minors = [p[2], ps[i][2]]
    inters = [p[1], ps[i][1]]
    ax.plot(minors, majors, inters, color='black', alpha=0.5)
    ax.scatter(ps[i][2], ps[i][0], ps[i][1], color='red', marker='1')
    return ps[i]


n_points = 5
majors, inters, minors = points(min_c, max_c, n_points)
for i in range(n_points):
    p0 = [majors[i], inters[i], minors[i]]
    ax.scatter(minors[i], majors[i], inters[i], color='blue', marker='x')
    # major_failure(hoek_brown, [majors[i], inters[i], minors[i]], color='black', alpha=0.5)
    p1 = distance(hoek_brown, [majors[i], inters[i], minors[i]])
    lp = limit_point(p0, p1)
    if lp is None:
        lp = limit_point(p0, p1, reverse=True)
    if lp is None:
        lp = p0
    ax.scatter(lp[2], lp[0], lp[1], color='green', marker='2')
    ax.plot([p1[2], lp[2]], [p1[0], lp[0]], [p1[1], lp[1]], color='black', alpha=0.5)
    # ax.plot([min_c, max_c], [min_c, max_c], [min_c, max_c], color='black', alpha=0.5)
    # major_failure(hoek_brown, p, color='black', alpha=0.5)

# # Tension
# x = list()
# z = list()
# y = list()
# for i in range(len(cs_curve)):
#     if cs_curve[i] <= 0:
#         x.append(cs_curve[i])
#         z.append(cs_curve[i])
#         y.append(g / 10 * cs_curve[i] + g)
# cut_plot(x, y, z, color='orange')
# for j in range(20):
#     x = list()
#     z = list()
#     y = list()
#     for i in range(len(cs_curve)):
#         if cs_curve[i] <= 0:
#             x.append(cs_curve[i])
#             z.append(cs_curve[i] + j)
#             y.append(g / 10 * (x[i] - (z[i] - x[i])) + g)
#     cut_plot(x, y, z, color='orange', alpha=0.2)

ax.set_xlabel('minor')
ax.set_ylabel('major')
ax.set_zlabel('inter')
ax.set_xlim([min_c, max_c])
ax.set_ylim([min_c, max_c])
ax.set_zlim([min_c, max_c])
# ax.legend()
plt.show()
