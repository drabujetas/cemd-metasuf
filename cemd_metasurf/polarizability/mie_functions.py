# Author: Diego Romero Abujetas, March 2023, diegoromeabu@gmail.com
#
# This file contain all the functions needed to calculate the polarizability of 
# Mie particles
#
# Calculation of (dipolar) polarizability of spheres by Mie theory.
# - sph_jn, sph_yn, sph_djn, sph_dyn,   (Spherical Bessel functions of integer order)
# - psi, diff_psi, xi, diff_xi          (Auxiliary functions)
# - Mie_an, Mie_bn              (Mie coefficients)
# - get_alpha                   (dipolar polarizability)

import numpy as np

def sph_jn(n,x):
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
    if n == 0:
        return np.cos(x)/x - np.sin(x)/x**2
    elif n > 0:
        return sph_jn(n-1,x) - (n+1)/x*sph_jn(n,x)
    else:
        return print("negative n are not implemented")

def sph_dyn(n,x):
    if n == 0:
        return np.cos(x)/x**2 + np.sin(x)/x
    elif n > 0:
        return sph_yn(n-1,x) - (n+1)/x*sph_yn(n,x)
    else:
        return print("negative n are not implemented")

def psi(n, x):
    return x * sph_jn(n,x)
    """
    if n == 0:
        return np.sin(x)
    elif n == 1:
        return np.sin(x)/x - np.cos(x)
    elif n > 1:
        psi_n = (2*n-1)/x*psi(n-1,x) - psi(n-2,x)
        return psi_n
    else:
        return print("negative n are not implemented")
    #return x * sps.spherical_jn(n, x, 0)
    """

def diff_psi(n, x):
    return sph_jn(n, x) + x * sph_djn(n, x)

def xi(n, x):
    return x * (sph_jn(n,x) + 1j*sph_yn(n,x))
    """
    if n == 0:
        return np.sin(x) - 1j*np.cos(x)
    elif n == 1:
        return (np.sin(x)/x - np.cos(x)) - 1j*(np.cos(x)/x + np.sin(x))
    elif n > 1:
        xi_n = (2*n-1)/x*xi(n-1,x) - xi(n-2,x)
        return xi_n
    else:
        return print("negative n are not implemented")
    #return x * (sps.spherical_jn(n, x, 0) + 1j * sps.spherical_yn(n, x, 0))
    """

def diff_xi(n, x):
    return (sph_jn(n, x) + 1j * sph_yn(n, x)) + x * (sph_djn(n, x) + 1j * sph_dyn(n, x))


# Mie_n: Function that calculates the Mie coefficients.
#
# Inputs:
#
# "k0" is wavevector in vacuum.
#
# "R" is the particle radius.
#
# "m_p" is the particle refractive index.
#
# "m_bg" is the background refractive index.
#
# "order" is the harmonic number order (integer number).
#
# Outpus:
#
# "an" and "bn" are the Mie coefficients or order "n".

def mie_n(k0, r_p, m_p, m_bg, order):

    alpha = k0 * r_p * m_bg
    beta = k0 * r_p * m_p
    mt = m_p / m_bg

    an = (mt * diff_psi(order, alpha) * psi(order, beta) - psi(order, alpha) * diff_psi(order,beta)) / (mt * diff_xi(order, alpha) * psi(order, beta) - xi(order, alpha) * diff_psi(order, beta))
    bn = (mt * psi(order, alpha) * diff_psi(order, beta) - diff_psi(order, alpha) * psi(order,beta)) / (mt * xi(order, alpha) * diff_psi(order, beta) - diff_xi(order, alpha) * psi(order, beta))

    return an, bn


# "get_alpha_Mie" calculates the polarizability using Mie theory.
#
# Inputs: 
#
# "ko" is wavevector in vacuum.
#
# "R_p" is the particle radius.
#
# "n_p" is the particle refractive index.
#
# "n_b" is the background refractive index.
#
# Outpus:
#
# "alp" is a "6 by 6" matrix with the polarizability of the particle

def get_alpha_mie(k0, r_p, n_p, n_b):

    k = k0*n_b

    a1, b1 = mie_n(k0, r_p, n_p, n_b, 1)
    alpha_e = 1j*(6*np.pi)/(k**3)*a1
    alpha_m = 1j*(6*np.pi)/(k**3)*b1 
    id3 = np.eye(3)
    alp = np.zeros((6,6), dtype = 'complex_')  
    alp[0:3,0:3] = id3*alpha_e
    alp[3:6,3:6] = id3*alpha_m

    return alp   