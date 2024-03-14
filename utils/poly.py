import numpy as np
import numba as nb

@nb.jit('f8[:,:](f8[:], i8)')
def _coeff_mat(x, deg):
    mat_ = np.zeros(shape=(x.shape[0],deg + 1), dtype='f8')
    const = np.ones_like(x)
    mat_[:,0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_

@nb.jit('f8[:](f8[:,:], f8[:])')
def _fit_x(a, b):
    # linalg solves ax = b
    det_ = np.linalg.lstsq(a, b)[0]
    return det_

@nb.jit('f8[:](f8[:], f8[:], i8)')
def _fit_poly(x, y, deg):
    a = _coeff_mat(x, deg)
    p = _fit_x(a, y)
    # Reverse order so p[0] is coefficient of highest order
    return p[::-1]

@nb.jit('f8(f8[:], f8[:])')
def _eval_polynomial(P: np.ndarray, x) -> float:
    '''
    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0].  Uses Horner's Method.
    '''
    result = np.zeros_like(x)
    for coeff in P:
        result = x * result + coeff
    fitting_error = np.sum((result - x) ** 2)
    return fitting_error

class Polynomials:
    def fit_poly(x, y, deg):
        coeffs = _fit_poly(x, y, deg)
        return coeffs, _eval_polynomial(coeffs, x)