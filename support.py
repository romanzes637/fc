import math


def distance(p0, p1):
    delta = list(map(lambda x0, x1: x1 - x0, p0, p1))
    norm = math.sqrt(sum([x * x for x in delta]))
    return norm


def angle(p0, p1):
    dot = sum(map(lambda x0, x1: x0 * x1, p0, p1))
    norm0 = math.sqrt(sum([x * x for x in p0]))
    norm1 = math.sqrt(sum([x * x for x in p1]))
    a = math.acos(dot / (norm0 * norm1))
    return a


def delta(p0, p1):
    d = list(map(lambda x0, x1: x1 - x0, p0, p1))
    return d


def cos(p0, p1):
    d = dot(p0, p1)
    n0 = norm(p0)
    n1 = norm(p1)
    if n0 == 0 or n1 == 0:
        # print('Warning! Zero norm!')
        c = 1
    else:
        c = d / (n0 * n1)
    return c


def dist(p0, p1):
    return norm(summ(p1, [-x for x in p0]))


def summ(p0, p1):
    return list(map(lambda x0, x1: x0 + x1, p0, p1))


def subs(p0, p1):
    return list(map(lambda x0, x1: x1 - x0, p0, p1))


def dot(p0, p1):
    d = sum(list(map(lambda x0, x1: x0 * x1, p0, p1)))
    return d


def norm(p0):
    return math.sqrt(sum(map(lambda x: x * x, p0)))


def unit(p0):
    n = norm(p0)
    if n != 0:
        u = [x / n for x in p0]
    else:
        u = 0
    return u


def projection(p0, p1, p2):
    r = subs(p0, p1)
    u = unit(r)
    d = dot(r, p2)
    n = norm(r)
    if n != 0:
        p = d / n
    else:
        p = 0
    if d != 0:
        p3 = [p * x for x in u]
    else:
        p3 = [0, p2[1]]
    return p3


def intersection(p0, p1, p2, p3):
    if p1[0] - p0[0] != 0 and p3[0] - p2[0] != 0:
        k1 = (p1[1] - p0[1]) / (p1[0] - p0[0])
        k2 = (p3[1] - p2[1]) / (p3[0] - p2[0])
        b1 = p0[1] - k1 * p0[0]
        b2 = p2[1] - k2 * p2[0]
        if k1 == k2:
            p5 = None
        else:
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
            p5 = [x, y]
    elif p1[0] - p0[0] == 0 and p3[0] - p2[0] != 0:
        k2 = (p3[1] - p2[1]) / (p3[0] - p2[0])
        b2 = p2[1] - k2 * p2[0]
        x = p0[0]
        y = k2 * x + b2
        p5 = [x, y]
    elif p1[0] - p0[0] != 0 and p3[0] - p2[0] == 0:
        k1 = (p1[1] - p0[1]) / (p1[0] - p0[0])
        b1 = p0[1] - k1 * p0[0]
        x = p2[0]
        y = k1 * x + b1
        p5 = [x, y]
    else:
        p5 = None
    return p5


def line(p0, p1):
    if (p1[0] - p0[0]) != 0:
        k = (p1[1] - p0[1]) / (p1[0] - p0[0])
        b = p0[1] - k * p0[0]
        return k, b
    else:
        return None