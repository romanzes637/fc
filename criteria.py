import math


def hoek_brown_mean_stresses(sigma_mean, c0, t0):
    """
    Hoek-Brown failure criterion in terms of mean stresses
    https://en.wikipedia.org/wiki/Hoek-Brown_failure_criterion
    https://link.springer.com/article/10.1007/s00603-012-0276-4
    :param float sigma_mean: mean normal stress
    (maximum principal stress + minimum principal stress) / 2
    https://en.wikipedia.org/wiki/Mohr-Coulomb_theory
    :param float c0: uniaxial compressive strength
    :param float t0: uniaxial tensile strength
    :return: maximum mean shear stress (two variants)
    (maximum principal stress - minimum principal stress) / 2
    https://en.wikipedia.org/wiki/Mohr-Coulomb_theory
    :rtype: tuple of float
    """
    a = (c0 * c0 - t0 * t0) / t0
    b = c0
    min_sigma_mean = (- a * a - 16 * b * b) / (16 * a)
    if sigma_mean < min_sigma_mean:
        tau_mean_1 = -1000
    else:
        tau_mean_1 = (-a + math.sqrt(a * a + 16 * (a * sigma_mean + b * b))) / 8
        # print(e)
        # print(sigma_mean)
    # try:
    #     tau_mean_2 = (-a - math.sqrt(a * a + 16 * (a * sigma_mean + b * b))) / 8
    # except ValueError as e:
    #     tau_mean_2 = None
    #     print(e)
    #     print(sigma_mean)
    # return tau_mean_1, tau_mean_2
    return tau_mean_1


def mohr_coulomb_mean_stresses(sigma_mean, c, phi):
    """
    Mohr-Coulomb failure criterion in terms of mean stresses
    https://en.wikipedia.org/wiki/Mohr-Coulomb_theory
    :param float sigma_mean: mean normal stress
    (maximum principal stress + minimum principal stress) / 2
    :param float c: cohesion
    :param phi: angle of internal friction
    :return: maximum mean shear stress
    (maximum principal stress - minimum principal stress) / 2
    :rtype: float
    """
    tau_mean = sigma_mean * math.sin(phi) + c * math.cos(phi)
    return tau_mean


def hoek_brown(sigma_2, sigma_3, mi=20, d=0.5, gsi=85, sigma_ci=200):
    # assert (sigma_2 > sigma_3)
    mb = mi * math.exp((gsi - 100) / (28 - 14 * d))
    s = math.exp((gsi - 100) / (9 - 3 * d))
    a = 0.5 + (math.exp(-gsi / 15) - math.exp(-20 / 3)) / 6
    if mb * sigma_3 / sigma_ci + s >= 0:
        sigma_1 = sigma_3 + sigma_ci * (mb * sigma_3 / sigma_ci + s) ** a
    else:
        sigma_1 = sigma_3
    if sigma_1 >= sigma_2 >= sigma_3:
        return sigma_1
    else:
        return max(sigma_2, sigma_3)
    # sigma_1 = sigma_3 + sigma_ci * (mb * sigma_3 / sigma_ci + s) ** a
    # return sigma_1


def mises(s):
    assert (s[0] >= s[1] >= s[2])
    st = math.sqrt(
        ((s[0] - s[1]) ** 2 + (s[1] - s[0]) ** 2 + (s[2] - s[0]) ** 2)) / 2
    return st


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    c0 = 100
    t0 = 10
    c = 30
    phi = 0.785
    min_sigma_mean = -50
    max_sigma_mean = 300
    n_sigma_mean = 351
    sigma_means = np.linspace(min_sigma_mean, max_sigma_mean, n_sigma_mean)
    tau_means_1 = [hoek_brown_mean_stresses(x, c0, t0) for x in sigma_means]
    # tau_means_2 = [hoek_brown_mean_stresses(x, c0, t0)[1] for x in sigma_means]
    tau_means_3 = [mohr_coulomb_mean_stresses(x, c, phi) for x in sigma_means]
    plt.plot(sigma_means, tau_means_1, label='Hoek-Brown 1')
    # plt.plot(sigma_means, tau_means_2, label='Hoek-Brown 2')
    plt.plot(sigma_means, tau_means_3, label='Mohr-Coulomb')
    plt.legend()
    plt.xlabel('sigma mean')
    plt.ylabel('tau mean')
    plt.show()
