import math

from matplotlib.colors import BoundaryNorm

from hb import hoek_brown_min_point
from support import cos, dist, subs, line


def hb_intersection(p0, p1, rtol=0.01, maxit=100, mi=20.0, d=0.5, gsi=85.0, sigma_ci=200.0):
    result = line(p0, p1)
    if result is not None:
        k = result[0]
        b = result[1]
    else:
        x = 0
        y = hoek_brown(x, x, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        return x, y
    if k != 1:
        xy = b / (1 - k)
    else:
        xy = 0
    s1min, s2min, s3min = hoek_brown_min_point(mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
    if xy < s3min:
        return xy, xy
    x = s3min
    dx = xy - s3min
    rdy = 1
    i = 0
    while rdy > rtol and i < maxit:
        i += 1
        x += dx
        hb_y = hoek_brown(x, x, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        pp_y = k * x + b
        dy = hb_y - pp_y
        if dy > 0:
            x -= dx
            dx *= 0.5
            continue
        length = dist([xy, xy], [pp_y, x])
        rdy = abs(dy) / length
    y = k * x + b
    return x, y


def hb_normal(p, cos_tol=0.001, maxit=1000, mi=20.0, d=0.5, gsi=85.0, sigma_ci=200.0):
    s1min, s2min, s3min = hoek_brown_min_point(mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
    x = s3min
    y = s1min
    i = 0
    c = 1
    delta_x = p[1] - s3min
    kdx = 1e-6
    hb1 = hoek_brown(p[0], p[0], mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
    if p[1] <= s1min:
        dx = kdx
        dy = hoek_brown(s3min + dx, s3min + dx, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci) - y
        p_hb = subs([x, y], p)
        c = cos(p_hb, [dx, dy])
        return [x, y], abs(c), i
    elif hb1 == p[1]:
        x = p[0]
        y = p[1]
        return [x, y], 0, i
    while abs(c) > cos_tol and i < maxit:
        i += 1
        x += delta_x
        y = hoek_brown(x, x, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        dx = delta_x * kdx
        dy = hoek_brown(x + dx, x + dx, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci) - y
        p_hb = subs([x, y], p)
        c = cos(p_hb, [dx, dy])
        if c < 0:
            x -= delta_x
            delta_x *= 0.5
    return [x, y], abs(c), i


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    from criteria import hoek_brown

    mi = 30
    mis = [30]
    d = 0.8
    gsi = 80
    sigma_ci = 100.0
    # Variations
    nc = 11
    min_mi = 20  # 4
    max_mi = 30  # 33
    # min_sigma_ci = 1  # 0
    # max_sigma_ci = 250  # 250
    # min_d = 0  # 0
    # max_d = 1  # 1
    # min_gsi = 5  # 5
    # max_gsi = 100  # 100
    # mis = np.linspace(min_mi, max_mi, nc)
    # sigma_cis = np.linspace(min_sigma_ci, max_sigma_ci, nc)
    # ds = np.linspace(min_d, max_d, nc)
    # gsis = np.linspace(min_gsi, max_gsi, nc)
    # min_color = 0.3
    for mi in mis:
        # HB
        mb = mi * math.exp((gsi - 100) / (28 - 14 * d))
        s = math.exp((gsi - 100) / (9 - 3 * d))
        a = 0.5 + (math.exp(-gsi / 15) - math.exp(-20 / 3)) / 6
        min_c = -500
        max_c = 500
        n = 20001
        xs = [min_c, max_c]
        ys = [min_c, max_c]
        hb1min, hb2min, hb3min = hoek_brown_min_point(mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        hb_xs = np.linspace(hb3min, max_c, n)
        hb_ys = [hoek_brown(x, x, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci) for x in hb_xs]
        plt.plot(hb_xs, hb_ys, color='black', linewidth=2, label='HB, mi={:.2f}, D={:.2f}, GSI={:.0f},\nσci={:.0f} МПа,mb={:.2f}, s={:.2f}, a={:.2f}'.format(mi, d, gsi, sigma_ci, mb, s, a, ))
        # plt.plot(hb_xs, hb_ys, color='black', linewidth=2, label='HB, mb={:.2f}, s={:.2f},\na={:.2f}, σci={:.0f} МПа'.format(mb, s, a, sigma_ci))
        # Points
        sx = [-100, 200]
        sp = [200, 300]
        smin = [0, 0]
        # smax2 = [sp[0], hoek_brown(sp[0], sp[0], mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)]
        # smin3 = projection([0, 0], [1, 1], sp)
        # smin2 = [sp[0], sp[0]]
        # smin4 = [sp[1], sp[1]]
        # smax5, c, cnt = hb_normal(sp, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        # smax5sx, csx, cntsx = hb_normal(sx, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        # smin5 = intersection(smax5, sp, [0, 0], [1, 1])
        # smax3 = hb_intersection(smin3, sp, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        # smax4 = hb_intersection(smin4, sp, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        # smax6 = hb_intersection(smin, sx, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)
        # plt.scatter(*sx, c='purple', marker='x', label='разрушен', s=50**2)
        # plt.scatter(*sp, c='blue', marker='+', label='не разрушен')
        # plt.scatter(*sp, c='black', marker='+', label='σ3+, σ1+')
        # plt.scatter(*sx, c='black', marker='x', label='σ3x, σ1x')
        # plt.scatter(*smax, c='black', marker='X', label='σ3x=σ3+, σ1max=HB(σ3x)')
        # plt.scatter(*smin, c='black', marker='P')
        # plt.scatter(*smin2, c='black', marker='P')
        # plt.scatter(*smin3, c='black', marker='P')
        # plt.scatter(*smin4, c='black', marker='P')
        # plt.scatter(*smin5, c='black', marker='P')
        # plt.scatter(*smax5, c='purple', marker='X')
        # plt.scatter(*smax6, c='purple', marker='X')
        nps = 51
        X, Y = np.meshgrid(np.linspace(min_c, max_c, nps), np.linspace(min_c, max_c, nps))
        ps = list(zip(np.ravel(X), np.ravel(Y)))
        # Norm
        norm_ps = np.array([hb_normal(x, mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)[0] for x in ps])
        data = list()
        for i, _ in enumerate(ps):
            p_to_norm = dist(ps[i], norm_ps[i])
            zero_to_p = dist(smin, ps[i])
            zero_to_norm = dist(smin, norm_ps[i])
            inter = hb_intersection(smin, ps[i])
            zero_to_inter = dist(smin, inter)
            # proj = projection([0, 0], [1, 1], ps[i])
            # proj_to_p = dist(proj, ps[i])
            # nproj = intersection(norm_ps[i], ps[i], [0, 0], [1, 1])
            # if nproj is not None:
            #     nproj_to_p = dist(nproj, ps[i])
            # else:
            #     nproj_to_p = 0
            if ps[i][1] >= hb1min:
                if ps[i][1] < norm_ps[i][1]:
                    c1 = zero_to_p / (zero_to_p + p_to_norm)
                else:
                    # c1 = zero_to_p / zero_to_inter
                    c1 = (zero_to_norm + p_to_norm) / zero_to_norm
            else:
                # c1 = zero_to_p / zero_to_inter
                c1 = (zero_to_norm + p_to_norm) / zero_to_norm
            data.append(c1)
        zs = np.array(data)
        # Contour
        # zs = np.array(
        #     [hb_normal([x, y], mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)[2] for x, y in zip(np.ravel(X), np.ravel(Y))])
        mask = np.array([x[0] > x[1] for x in ps])
        Z = zs.reshape(X.shape)
        Z = np.ma.array(Z, mask=mask)
        # # im = plt.imshow(Z, cmap=cm.get_cmap('Greys'), interpolation='bilinear', origin='lower', extent=(-100, 300, -100, 300))
        # # levels = np.percentile(Z, np.linspace(0, 100, 101))
        levels = np.concatenate((np.linspace(0, 1, 21), np.linspace(2, 10, 9), np.linspace(20, math.ceil(max(zs)), 12)))
        # levels = np.concatenate((np.linspace(0, 1, 11), np.linspace(2, math.ceil(max(zs)), 11)))
        b_norm = BoundaryNorm(levels, 256)
        csf = plt.contourf(X, Y, Z, norm=b_norm, levels=levels, corner_mask=True, cmap=cm.get_cmap('RdYlGn_r'))
        # cs = plt.contour(X, Y, Z, corner_mask=True, cmap=cm.get_cmap('viridis'))
        # plt.clabel(cs, cs.levels, inline=True, fontsize=25)
        # plt.contour(cs)
        cb_label = 'Поврежденность = a/(a+b) или (c+d)/d\n ' \
                   'a - расстояние от точки минимума до текущей точки,\n' \
                   'b - расстояние от текущей точки до кривой разрушения,\n' \
                   'c - расстояние от кривой разрушения до текушей точки,\n' \
                   'd - расстояние от точки минимума до кривой разрушения'
        plt.colorbar(csf, label=cb_label, ticks=levels, format='%.2f')
        # Lines
        # plt.plot(*zip(smin, sx), label='0 -> σx', color='purple', linestyle='--')
        # plt.plot(*zip(smin2, sp), label='σ3, σ1=σ3')
        # plt.plot(*zip(smax2, sp), label='σ3, HB(σ3)')
        # plt.plot(*zip(smin3, sp), label='normal from σ to σ1=σ3')
        # plt.plot(*zip(smax3, sp), label='intersection of HB with normal from σ to σ1=σ3')
        # plt.plot(*zip(smin4, sp), label='σ3=σ1, σ1')
        # plt.plot(*zip(smax4, sp), label='HB-1(σ1), σ1')
        # plt.plot(*zip(smin5, sp), label='intersection of normal from σ to HB with σ1=σ3')
        # plt.plot(*zip(smin, sp), label='a - 0 до σ+', color='blue')
        # plt.plot(*zip(smax5, sp), label='b - нормаль σ+ к HB', color='blue', linestyle='--')
        # plt.plot(*zip(smax5sx, sx), label='c - нормаль σx к HB', color='purple', linestyle='--')
        # plt.plot(*zip(smin, smax5sx), label='d - 0 до нормали σx к HB', color='purple')
        # plt.plot(*zip(smin, smax5), label='0 -> HB+', color='blue')
        # plt.text(*sx, s='σx', fontsize=12, color='purple')
        # plt.text(*[0.5*(smax5sx[0] + sx[0]), 0.5*(smax5sx[1] + sx[1])], s='c', fontsize=12, color='purple')
        # plt.text(*[0.5*(smax5sx[0] + smin[0]), 0.5*(smax5sx[1] + smin[1])], s='d', fontsize=12, color='purple')
        # plt.text(*[0.5*(smin[0] + sp[0]), 0.5*(smin[1] + sp[1])], s='a', fontsize=12, color='blue')
        # plt.text(*[0.5*(smax5[0] + sp[0]), 0.5*(smax5[1] + sp[1])], s='b', fontsize=12, color='blue')
        # plt.text(*smax5sx, s='σxn', fontsize=12, color='purple')
        # plt.text(*smax5, s='σ+n', fontsize=12, color='blue')
        # plt.text(*[25, -25], s='0', fontsize=12, color='black', verticalalignment='center', horizontalalignment='center')
        # plt.text(*sp, s='σ+', fontsize=12, color='blue')
        # General
        s1s3_xs = np.linspace(min_c, max_c, n)
        plt.plot(s1s3_xs, s1s3_xs, color='black', linewidth=1, linestyle=':', label='σ1=σ3')
        plt.xlim(min_c, max_c)
        plt.ylim(min_c, max_c)
        plt.xticks(np.linspace(min_c, max_c, 11))
        plt.yticks(np.linspace(min_c, max_c, 11))
        plt.xlabel('σ3, МПа')
        plt.ylabel('σ1, МПа')
        plt.plot([min_c, max_c], [0, 0], color='black', linewidth=0.5)
        plt.plot([0, 0], [min_c, max_c], color='black', linewidth=0.5)
        plt.grid()
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig('mi_{}.png'.format(mi))
        plt.show()
        plt.clf()
