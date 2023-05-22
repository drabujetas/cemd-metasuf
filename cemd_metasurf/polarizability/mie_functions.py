"""
The file mie_functions.py contain all the functions needed to calculate the (dipolar) polarizability of Mie spherical particles.

List of functions: 
    - sph_jn, sph_yn, sph_djn, sph_dyn,   (Spherical Bessel functions of integer order)
    - psi, diff_psi, xi, diff_xi          (Auxiliary functions)
    - Mie_an, Mie_bn              (Mie coefficients)
    - get_alpha                   (dipolar polarizability)
"""
import numpy as np

def sph_jn(n,x):
    """
    Spherical Bessel function of the first kind.

    :param n: Order
    :type n: int
    :param x: Argument
    :type x: float

    :return: Spherical Bessel function of the first kind.
    """
    if n == 0:
        return np.sin(x)/x
    elif n == 1:
        return np.sin(x)/x**2 - np.cos(x)/x
    elif n > 1:
        j_n = (2*n-1)/x*sph_jn(n-1,x) - sph_jn(n-2,x)
        return j_n
    else:
        return print("negative n are not implemented")   

def sph_yn(n,x):
    """
    Spherical Bessel function of the second kind.

    :param n: Order
    :type n: int
    :param x: Argument
    :type x: float

    :return: Spherical Bessel function of the second kind.
    """
    if n == 0:
        return -np.cos(x)/x
    elif n == 1:
        return -np.cos(x)/x**2 - np.sin(x)/x
    elif n > 1:
        y_n = (2*n-1)/x*sph_yn(n-1,x) - sph_yn(n-2,x)
        return y_n
    else:
        return print("negative n are not implemented")  

def sph_djn(n,x):
    """
    Derivative of the spherical Bessel function of the first kind.

    :param n: Order.
    :type n: int
    :param x: Argument.
    :type x: float

    :return: Derivative of the spherical Bessel function of the first kind.
    """
    if n == 0:
        return np.cos(x)/x - np.sin(x)/x**2
    elif n > 0:
        return sph_jn(n-1,x) - (n+1)/x*sph_jn(n,x)
    else:
        return print("negative n are not implemented")

def sph_dyn(n,x):
    """
    Derivative of the spherical Bessel function of the second kind.

    :param n: Order.
    :type n: int
    :param x: Argument.
    :type x: float

    :return: Derivative of the spherical Bessel function of the second kind.
    """
    if n == 0:
        return np.cos(x)/x**2 + np.sin(x)/x
    elif n > 0:
        return sph_yn(n-1,x) - (n+1)/x*sph_yn(n,x)
    else:
        return print("negative n are not implemented")

def psi(n, x):
    """
    Auxiliary function for Mie coeficients.

    :param n: Order.
    :type n: int
    :param x: Argument.
    :type x: float

    :return: Auxiliary function.
    """
    return x * sph_jn(n,x)

def diff_psi(n, x):
    """
    Auxiliary function for Mie coeficients.

    :param n: Order.
    :type n: int
    :param x: Argument.
    :type x: float

    :return: Auxiliary function.
    """
    return sph_jn(n, x) + x * sph_djn(n, x)

def xi(n, x):
    """
    Auxiliary function for Mie coeficients.

    :param n: Order.
    :type n: int
    :param x: Argument.
    :type x: float

    :return: Auxiliary function.
    """
    return x * (sph_jn(n,x) + 1j*sph_yn(n,x))

def diff_xi(n, x):
    """
    Auxiliary function for Mie coeficients.

    :param n: Order.
    :type n: int
    :param x: Argument.
    :type x: float

    :return: Auxiliary function.
    """
    return (sph_jn(n, x) + 1j * sph_yn(n, x)) + x * (sph_djn(n, x) + 1j * sph_dyn(n, x))

def mie_n(k0, r_p, m_p, m_bg, order):
    """
    Calculation of Mie coeficients.

    :param k0: Vacuum wavevector.
    :type k0: float
    :param r_p: Particle radius.
    :type r_p: float
    :param m_p: Particle refractive index.
    :type m_p: float
    :param m_bg: Background refractive index.
    :type m_bg: float
    :param n: Order.
    :type n: int

    :return:
        - The first element is the electric Mie coefficient, an.
        - The second element is the magnetic Mie coefficient, bn.
    """
    alpha = k0 * r_p * m_bg
    beta = k0 * r_p * m_p
    mt = m_p / m_bg

    an = (mt * diff_psi(order, alpha) * psi(order, beta) - psi(order, alpha) * diff_psi(order,beta)) / (mt * diff_xi(order, alpha) * psi(order, beta) - xi(order, alpha) * diff_psi(order, beta))
    bn = (mt * psi(order, alpha) * diff_psi(order, beta) - diff_psi(order, alpha) * psi(order,beta)) / (mt * xi(order, alpha) * diff_psi(order, beta) - diff_xi(order, alpha) * psi(order, beta))

    return an, bn

def get_alpha_mie(k0, r_p, m_p, m_bg):
    """
    Calculation of the dipolar polarizability using the Mie coeficients.

    :param k0: Vacuum wavevector.
    :type k0: float
    :param r_p: Particle radius.
    :type r_p: float
    :param m_p: Particle refractive index.
    :type m_p: float
    :param m_bg: Background refractive index.
    :type m_bg: float

    :return: 6x6 polarizability matrix alp.
    """
    k = k0*m_bg

    a1, b1 = mie_n(k0, r_p, m_p, m_bg, 1)
    alpha_e = 1j*(6*np.pi)/(k**3)*a1
    alpha_m = 1j*(6*np.pi)/(k**3)*b1 
    id3 = np.eye(3)
    alp = np.zeros((6,6), dtype = 'complex_')  
    alp[0:3,0:3] = id3*alpha_e
    alp[3:6,3:6] = id3*alpha_m

    return alp   
