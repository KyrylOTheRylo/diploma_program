from scipy.integrate import quad, dblquad
import numpy as np
from numba import njit


@njit
def x_power_2_plus_t_power_2(x, t):
    return x ** 2 + t ** 2


def test_db_quad(expr= x_power_2_plus_t_power_2):
    return dblquad(expr, 0, 1, lambda x: 0, lambda x: 2, epsabs=1e-6, epsrel=1e-6)


print(test_db_quad())