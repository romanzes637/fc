import math
import unittest


from support import distance, angle, delta, cos, subs, dist


def norm(func, kwargs, key_x, point, atol=0.01, maxit=100):
    """
    Normal point by distance and angle between tangent and normal
    :param function func: failure criterion function
    :param dict kwargs: func key-value arguments
    :param string key_x: key at kwargs for x coordinate
    :param tuple of float point: point in space
    :param float atol: absolute delta cos tolerance
    :param int maxit: maximum number of iterations
    :return: normal point at failure criterion curve
    :rtype: tuple of float
    """
    dx = 0.001
    delta_x = 1
    delta_angle = 100
    cnt = 0
    # a = 1
    x = kwargs[key_x]
    y = func(**kwargs)
    norm_point = (x, y)
    while delta_angle > atol and cnt < maxit:
        cnt += 1
        x = kwargs[key_x]
        y = func(**kwargs)
        # Move pos
        pos_x = x + delta_x
        kwargs[key_x] = pos_x
        pos_y = func(**kwargs)
        pos_point = (pos_x, pos_y)
        pos_tang = (pos_y - y, dx)
        # Move neg
        neg_x = x - delta_x
        kwargs[key_x] = neg_x
        neg_y = func(**kwargs)
        neg_point = (neg_x, neg_y)
        neg_tang = (neg_y - y, dx)
        # Distance
        pos_d = distance(point, pos_point)
        neg_d = distance(point, neg_point)
        print(pos_d, neg_d)
        if pos_d < neg_d:
            kwargs[key_x] = pos_x
            norm_point = pos_point
            tang = pos_tang
        elif pos_d > neg_d:
            kwargs[key_x] = neg_x
            norm_point = neg_point
            tang = neg_tang
        else:
            break
        # Angle
        a = angle(norm_point, tang)
        delta_angle = abs(a - math.pi / 2)
    print(delta_angle, cnt)
    return norm_point


def norm_cos(func, kwargs, key_x, start_x, point, atol=0.001, maxit=1000):
    """
    Normal point by cos between tangent and normal
    :param function func: failure criterion function
    :param dict kwargs: func key-value arguments
    :param string key_x: key at kwargs for x coordinate
    :param tuple of float point: point in space
    :param float atol: absolute delta cos tolerance
    :param int maxit: maximum number of iterations
    :return: normal point at failure criterion curve
    :rtype: tuple of float
    """
    c = atol + 1
    delta_x = 1000
    kwargs[key_x] = start_x
    kdx = 1e-6
    cnt = 0
    norm_x = kwargs[key_x]
    norm_y = func(**kwargs)
    norm_point = (norm_x, norm_y)
    while abs(c) > atol and cnt < maxit:
        cnt += 1
        x = kwargs[key_x]
        # Move
        norm_x = x + delta_x
        kwargs[key_x] = norm_x
        norm_y = func(**kwargs)
        norm_point = (kwargs[key_x], norm_y)
        # Tangent
        dx = delta_x * kdx
        tang_x = norm_x + dx
        kwargs[key_x] = tang_x
        tang_y = func(**kwargs)
        dy = tang_y - norm_y
        tang = (dx, dy)
        norm = subs(norm_point, point)
        c = cos(norm, tang)
        kwargs[key_x] = x + delta_x
        # plt.scatter(*norm_point)
        if c < 0:
            kwargs[key_x] -= delta_x
            delta_x *= 0.5
    # print(c, i)
    return norm_point, c, cnt


def norm_sigma(sigma, atol=0.01, maxit=100):
    from criteria import hoek_brown

    sigma[1] = sigma[2]
    sigma_max_1 = hoek_brown(sigma[1], sigma[2])
    sigma_max_3 = sigma[2]
    sigma_max = [sigma_max_1, sigma[1], sigma[2]]
    if sigma_max_1 < sigma[0]:
        sigma_n = [hoek_brown(0, 0), 0.0, 0.0]
    else:
        sigma_n = sigma_max
    d = distance(sigma, sigma_n)
    dx = 1
    pd = d
    add = 100
    cnt = 0
    a = 1
    while add > atol and cnt < maxit:
        cnt += 1
        # Move pos and neg
        pos_sigma_n = [hoek_brown(sigma_n[1] + dx, sigma_n[2] + dx),
                       sigma_n[1] + dx, sigma_n[2] + dx]
        neg_sigma_n = [hoek_brown(sigma_n[1] - dx, sigma_n[2] - dx),
                       sigma_n[1] - dx, sigma_n[2] - dx]
        if pos_sigma_n[0] <= pos_sigma_n[2] or neg_sigma_n[0] <= neg_sigma_n[
            2]:
            # sigma_n = [0.0, 0.0, 0.0]
            break
        # New distances
        pos_d = distance(sigma, pos_sigma_n)
        neg_d = distance(sigma, neg_sigma_n)
        if pos_d < neg_d:
            sigma_n[1] += dx
            sigma_n[2] += dx
            sigma_n[0] = pos_sigma_n[0]
            d = pos_d
        elif pos_d > neg_d:
            sigma_n[1] -= dx
            sigma_n[2] -= dx
            sigma_n[0] = neg_sigma_n[0]
            d = neg_d
        else:
            break
        k = 1
        # cmap = cm.get_cmap('Set1')
        # c = cmap(random.uniform(0, 1))
        d_sigma_n = [
            hoek_brown(sigma_n[1] + k * dx, sigma_n[2] + k * dx) - sigma_n[0],
            k * dx, k * dx]
        # plt.plot([sigma_n[2], sigma_n[2] + d_sigma_n[2]], [sigma_n[0], sigma_n[0] + d_sigma_n[0]], color=c)
        ds = delta(sigma, sigma_n)
        # plt.plot([sigma[2], sigma[2] + ds[2]], [sigma[0], sigma[0] + ds[0]], color=c)
        a = angle(ds, d_sigma_n)
        dsdydx = ds[0] / ds[1]
        dsb = sigma[0] - dsdydx * sigma[1]
        dsb2 = sigma_n[0] - dsdydx * sigma_n[1]
        sigma_min_3 = dsb / (1 - dsdydx)
        sigma_min_1 = sigma_min_3
        # plt.plot([sigma[2], sigma[2] + ds[2]], [sigma[0], sigma[0] + ds[0]], color=c)
        # plt.plot([sigma_n[2], sigma_min_3], [sigma_n[0], sigma_min_1])
        add = abs(a - math.pi / 2)
        pd = d
    return sigma_n


class TestNorm(unittest.TestCase):

    def test_norm_cos(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from criteria import hoek_brown_mean_stresses

        # Hoek-Brown curve
        min_x = -400
        max_x = 400
        n = 1001
        hb_xs = np.linspace(min_x, max_x, n)
        c0 = 100
        t0 = 10
        a = (c0 * c0 - t0 * t0) / t0
        b = c0
        min_sigma_mean = (- a * a - 16 * b * b) / (16 * a)
        hb_ys = [hoek_brown_mean_stresses(x, c0, t0) for x in hb_xs]
        plt.plot(hb_xs, hb_ys, label='Hoek-Brown')
        # Point
        stress = [50, 30, 10]
        sigma_mean = (stress[0] + stress[2]) / 2
        tau_mean = (stress[0] - stress[2]) / 2
        point = (sigma_mean, tau_mean)
        plt.scatter(sigma_mean, tau_mean, label='point')
        # Norm point
        kwargs = {
            'c0': 100,
            't0': 10,
            'sigma_mean': None
        }
        start_x = min_sigma_mean
        key_x = 'sigma_mean'
        norm_point, c, cnt = norm_cos(hoek_brown_mean_stresses, kwargs, key_x, start_x, point)
        plt.scatter(*norm_point, label='norm point')
        plt.plot(*zip(norm_point, point), label='norm')
        # Plot
        plt.plot([min_x, max_x], [0, 0], color='black', linewidth=0.5)
        plt.plot([0, 0], [min_x, max_x], color='black', linewidth=0.5)
        plt.xlim(min_x, max_x)
        plt.ylim(min_x, max_x)
        plt.grid()
        plt.legend()
        plt.xlabel('σm, МПа')
        plt.ylabel('τm, МПа')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def test_norm_cos_many(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm
        from matplotlib import cm
        import numpy as np

        from criteria import hoek_brown_mean_stresses

        # Points
        min_c = -200
        max_c = 200
        nps = 101
        xs = np.linspace(min_c, max_c, nps)
        ys = np.linspace(min_c, max_c, nps)
        xx, yy = np.meshgrid(xs, ys)
        ps = list(zip(np.ravel(xx), np.ravel(yy)))
        # Data
        data = list()
        c0 = 100
        t0 = 10
        a = (c0 * c0 - t0 * t0) / t0
        b = c0
        min_sigma_mean = (- a * a - 16 * b * b) / (16 * a)
        kwargs = {
            'c0': c0,
            't0': t0,
            'sigma_mean': None,
        }
        key_x = 'sigma_mean'
        start_x = min_sigma_mean
        for i, p in enumerate(ps):
            # zero_p = (0, 0)
            zero_p = (p[0], 0)
            norm_p, c, cnt = norm_cos(hoek_brown_mean_stresses, kwargs, key_x,
                                      start_x, p)
            # hb_0 = hoek_brown_mean_stresses(0, c0, t0)
            p_to_norm_p = dist(p, norm_p)
            zero_to_p = dist(zero_p, p)
            zero_to_norm_p = dist(zero_p, norm_p)
            # inter = hb_intersection(zero_p, p)
            # zero_to_inter = dist(zero_p, inter)
            # proj = projection([0, 0], [1, 1], ps[i])
            # proj_to_p = dist(proj, ps[i])
            # nproj = intersection(norm_ps[i], ps[i], [0, 0], [1, 1])
            # if nproj is not None:
            #     nproj_to_p = dist(nproj, ps[i])
            # else:
            #     nproj_to_p = 0
            # if p[1] >= hb1min:
            # if p[0] > hb_0:
            if p[1] < norm_p[1]:
                # damage = 0
                # damage = c
                # damage = cnt
                damage = zero_to_p / (zero_to_p + p_to_norm_p)
            else:
                # damage = 1
                # damage = c
                # damage = cnt
                # c1 = zero_to_p / zero_to_inter
                damage = (zero_to_norm_p + p_to_norm_p) / zero_to_norm_p
            # else:
            #     if p[1] < norm_p[1]:
            #         damage = zero_to_p / (zero_to_p + p_to_norm_p)
            #     else:
            #         # c1 = zero_to_p / zero_to_inter
            #         damage = (zero_to_norm_p + p_to_norm_p) / zero_to_norm_p
            #     # c1 = zero_to_p / zero_to_inter
            #     c1 = (zero_to_norm_p + p_to_norm_p) / zero_to_norm_p
            data.append(damage)
        zs = np.array(data)
        # Plot
        # Contour
        # zs = np.array(
        #     [hb_normal([x, y], mi=mi, d=d, gsi=gsi, sigma_ci=sigma_ci)[2] for x, y in zip(np.ravel(X), np.ravel(Y))])
        # mask = np.array([x[0] > x[1] for x in ps])
        # zz = np.ma.array(zs, mask=mask)
        # # im = plt.imshow(Z, cmap=cm.get_cmap('Greys'), interpolation='bilinear', origin='lower', extent=(-100, 300, -100, 300))
        # # levels = np.percentile(Z, np.linspace(0, 100, 101))
        levels = np.concatenate((np.linspace(0, 1, 21), np.linspace(2, 10, 9),
                                 np.linspace(20, max([math.ceil(max(zs)), 31]),
                                             12)))
        levels = np.concatenate((np.linspace(0, 1, 21), np.linspace(1.1, 1.9, 9),
                                 np.linspace(2, max([math.ceil(max(zs)), 3.1]),
                                             12)))
        # levels = np.concatenate((np.linspace(0, 1, 11), np.linspace(2, math.ceil(max(zs)), 11)))
        bn = BoundaryNorm(levels, 256)
        zz = zs.reshape(xx.shape)
        # csf = plt.contourf(xx, yy, zz, cmap=cm.get_cmap('RdYlGn_r'))
        csf = plt.contourf(xx, yy, zz, norm=bn, levels=levels, cmap=cm.get_cmap('RdYlGn_r'))
        # cb_label = 'Поврежденность = a/(a+b) или (c+d)/d\n ' \
        #            'a - расстояние от точки минимума до текущей точки,\n' \
        #            'b - расстояние от текущей точки до кривой разрушения,\n' \
        #            'c - расстояние от кривой разрушения до текушей точки,\n' \
        #            'd - расстояние от точки минимума до кривой разрушения'
        # cb_label = 'Поврежденность = a/(a+b) или (c+d)/d'
        cb_label = 'Поврежденность'
        plt.colorbar(csf, label=cb_label, ticks=levels, format='%.2f')
        # plt.colorbar(csf, format='%.2f')
        hb_ys = [hoek_brown_mean_stresses(x, c0, t0) for x in xs]
        plt.plot(xs, hb_ys, label='Hoek-Brown')
        # Plot
        plt.plot([min_c, max_c], [0, 0], color='black', linewidth=0.5)
        plt.plot([0, 0], [min_c, max_c], color='black', linewidth=0.5)
        plt.xlim(min_c, max_c)
        plt.ylim(min_c, max_c)
        plt.grid()
        plt.legend()
        plt.xlabel('σm = (σ1 + σ3) / 2, МПа')
        plt.ylabel('τm = (σ1 - σ3) / 2, МПа')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('{}.png'.format('tau0'), dpi=300)
        plt.show()


if __name__ == '__main__':
    unittest.main()
