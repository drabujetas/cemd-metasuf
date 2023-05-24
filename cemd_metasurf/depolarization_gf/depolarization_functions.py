"""
The file depolarization_functions.py contain all the functions needed to calculate the depolariztion function and for the class DGreenFunction.

List of functions: 
    - calc_gb_1puc     (depolarization Green function for one partilce per unit cell)
    - calc_gb_1d_kx    (depolarization Green function for one dimensional arrays, like an array of cylinders)
    - calc_gb_ch      (depolarization Green function for a chain of particles)
    - polylog_2, polylog_3, clausen2 (implementation of polylog functions at the unit circle)
"""

import numpy as np

def calc_gb_1puc(my_metasurface, n_sum = 100):
    """
    Function that calculates the Green function for an 2D array with one particle per unit cell.

    The parameters "n_sum" and "n_l" (defined in the code) are the number of elements taken in the sum. For bigger "n_sum"
    the convergence is better, but be careful with "n_l". With "n_l = 4" is
    enough to get a good convergence, but for bigger "n_l" is necessary to take
    bigger "n_sum" to get convergence. See PRB 2020, D. R. Abuejetas et. al., 102, 125411 
    for more information about the convergence and the menaing of "n_l".

    For better performance at the calculation of the depolarization Green function, try to place "kx" and "ky" inside the first Brillouin zone.
    I need to implement a function for doing that.

    :param my_metasurface: Vacuum wavevector.
    :type my_metasurface: classes.Metasurface
    :param n_sum: Number of elements taken in the sum.
    :type n_sum: int

    :return: 6x6 deplarization matrix for 2 dimensional arrays, gb_2d.
    """
    a, b, th = my_metasurface.get_lattice()
    k, kx, ky = my_metasurface.get_bloch()

    # I need to implement the folding for any lattice geometry.
    #kx = kx - np.floor( (kx + np.pi/a)/(2*np.pi/a))*(2*np.pi/a)   # bring "kx" to the first Brilluoin zone for square arrays
    #ky = ky - np.floor( (ky + np.pi/b)/(2*np.pi/b))*(2*np.pi/b)   # bring "ky" to the first Brilluoin zone for square arrays

    n_l = int(np.floor( np.real(k + np.abs(kx))/(2*np.pi/a) ) + 3) # convergence parameter
    if n_l > 7:
        n_l = 7
        raise ValueError("a/lambda >> 1")

    gb_ch = calc_gb_ch(a,k,kx)
    gb_1d = calc_gb_1d_kx(n_sum,b*np.sin(th),k,ky,kx)

    for m in range(n_l-1):

        kxlp = kx - 2*np.pi/a*(m + 1)
        kxlm = kx + 2*np.pi/a*(m + 1)
        n_sum_m = n_sum*(m+1)
        gb_1d = gb_1d + calc_gb_1d_kx(n_sum_m,b*np.sin(th),k,ky - ((kxlp-kx)*np.cos(th)/np.sin(th)),kxlp) + calc_gb_1d_kx(n_sum_m,b*np.sin(th),k,ky - ((kxlm-kx)*np.cos(th)/np.sin(th)),kxlm)

    gb_2d = gb_ch + 1/a*(gb_1d)

    return gb_2d
    
def calc_gb_1d_kx(n_sum,b,k,ky,kx):
    """
    Function for calculating the depolarization Green function for an array of 1D cylinders
    or particles with translational symmetry along the "x" axis where the projection of the 
    wavevector is along the "x" axis.
    Therefore, the cylinder axis is along the "x" axis and they are periodically spaced along the "y" axis. 
    
    Also, the sums have been rewriten to performe a convergence better than "1/m^3".

    :param n_sum: Number of elements taken in the sum.
    :type n_sum: int
    :param b: Lattice constant along the y-axis.
    :type b: float
    :param k: Wavevector in the medium.
    :type k: float
    :param ky: Bloch wavevector (Floquet periodicity) along the y-axis.
    :type ky: float
    :param kx: Periodicity along the trnslational symmetry x-axis.
    :type kx: float

    :return: 6x6 deplarization matrix for 1 dimensional arrays, gb_1d. 
    """
    g_euler = 0.577215664901532860606512090082402431042
    zr_3 = 1.2020569031595942853997381615114499907 #zeta Riemann evaluated at z = 3 
    f3 = (b/(2*np.pi))**3*zr_3
    
    m = np.linspace(1,n_sum,n_sum)
             
    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(kp ** 2 - ky ** 2, dtype = 'complex_')
    
    km = 2*np.pi*m/b
    kym = ky - km 
    kymm = ky + km
    kzm = np.sqrt(kp ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(kp ** 2 - kymm ** 2, dtype = 'complex_')
    
    fxx3 = (kz**2 + 3*ky**2)
    fyy3 = (4*k**2*kz**2 + 12*k**2*ky**2 - 10*ky**2*kz**2 - 7*ky**4 - 3*kz**4)/4
    fzz3 = (4*k**2*kz**2 + 12*k**2*ky**2 - 6*ky**2*kz**2 - 5*ky**4 - kz**4)/4
    fxy3 = 2j*kp**2*ky
    
    sum1 = (1j*(1/(2*kz*b) - 1./4) + 1/(2*b)*(np.sum(1j/kzm + 1j/(kzmm) - 2/km - fxx3/km**3) + fxx3*f3) + 1/(2*np.pi) * (np.log(kp*b/(4*np.pi)) + g_euler))
    sum2 = -(1/k*(1j*ky/(2*kz*b) + 1j/(2*b)*(np.sum(kym/kzm + kymm/kzmm - fxy3/km**3) + fxy3*f3) - 1/(2*np.pi)*ky))
    
    gb_xx = kp ** 2/k ** 2 * sum1    
    gb_yy = (1j/(2*kz*b)*(1 - ky ** 2/k ** 2) - 1j/8*(1 + kx ** 2/k ** 2) + 1/(2*k ** 2*b)*(np.sum(1j*(k ** 2 - kym ** 2)/kzm 
         + 1j * (k ** 2 - kymm ** 2)/kzmm - 1/km*(k ** 2 + kx ** 2 - 2*km ** 2) - fyy3/km**3 ) + fyy3*f3 ) + 1/(4*np.pi*k ** 2)*(np.log(kp*b/(4*np.pi)) + g_euler )*(k ** 2 + kx ** 2)
         + 1/(8*np.pi*k ** 2)*(ky ** 2 - kz ** 2) + 1/6*np.pi/(k ** 2*b ** 2) )
    gb_zz = ( 1j/(2*kz*b)*(1 - kz ** 2/k ** 2) - 1j/8*(1 + kx ** 2/k ** 2) + 1/(2*k ** 2*b)*(np.sum(1j*(k ** 2 - kzm ** 2)/kzm 
         + 1j*(k ** 2 - kzmm ** 2)/kzmm - 1/km*(k ** 2 + kx ** 2 + 2*km ** 2) - fzz3/km**3 ) + fzz3*f3 ) + 1/(4*np.pi*k ** 2)*(np.log(kp*b/(4*np.pi)) + g_euler )*(k ** 2 + kx ** 2)
         + 1/(8*np.pi*k ** 2)*(kz ** 2 - ky ** 2) - 1/6*np.pi/(k ** 2*b ** 2) )
    gb_xy = kx/k * sum2
    gb_yz = (kx/k * sum1)
    gb_zx = -sum2
    
    gb_1d = np.array([[gb_xx, gb_xy, 0,0,0,-gb_zx],[gb_xy, gb_yy, 0, 0,0,gb_yz], [0,0,gb_zz, gb_zx,-gb_yz,0], [0,0,gb_zx,gb_xx,gb_xy,0] , [0,0,-gb_yz,gb_xy,gb_yy,0], [-gb_zx,gb_yz,0,0,0,gb_zz]]) 
    
    return gb_1d


# "Gb_Ch" calculates the depolarization Green function of a chain of particles align along the "x" axis
# oriented for the calculation of "Gb" of an two dimensional array.
#
# Inputs:
#
# "d" is the distance between particles.
# "k" is the wavevector in the medium (It can be complex).
# "kp" is the projection of the wavevector over the axis of the chain (the "x" axis) (It is real).
#
# Outputs:
#
# GbCh is the contribution of the chain to the depolarization Green function of the two dimensional array.

def calc_gb_ch(d,k,kp):
    """
    Function for calculating the depolarization Green function of a chain of particles align along the "x" axis
    oriented for the calculation of "Gb" of an two dimensional array.
    
    :param d: Lattice constant along the x-axis.
    :type d: float
    :param k: Wavevector in the medium.
    :type k: float
    :param kp: Projection of the wavevector over the axis of the chain.
    :type kp: float

    :return: 6x6 deplarization matrix for a chain of particles, gb_ch. 
    """
    arg_minus = np.exp(1j * (k-kp) * d)
    arg_plus = np.exp(1j * (k+kp) * d)

    l1m = - np.log(1 - arg_minus)#mp.polylog(1,np.exp(1j * (k-kp) * d))
    l1p = - np.log(1 - arg_plus)#mp.polylog(1,np.exp(1j * (k+kp) * d))
    l2m = polylog_2(arg_minus)
    l2p = polylog_2(arg_plus)
    l3m = polylog_3(arg_minus)
    l3p = polylog_3(arg_plus)
    
    fac = 1j/(4*d ** 3 * k ** 2 * np.pi)
    dk = d * k
    dk2 = d ** 2 * k ** 2
    
    gb_xx = -2*fac*(dk * (l2m + l2p ) + 1j*( l3m + l3p ))
    gb_yy = fac * ( - 1j * dk2 *( l1m + l1p ) + dk * ( l2m + l2p ) + 1j *( l3m +  l3p ))
    gb_yz = fac*( - 1j * dk2 * ( l1m - l1p ) + dk *( l2m - l2p ))
          
    gb_ch = np.zeros((6, 6), dtype = 'complex_' )

    gb_ch[0,0] = gb_xx
    gb_ch[1,1] = gb_yy
    gb_ch[2,2] = gb_yy
    gb_ch[3,3] = gb_xx
    gb_ch[4,4] = gb_yy
    gb_ch[5,5] = gb_yy

    gb_ch[1,5] = gb_yz
    gb_ch[2,4] = - gb_yz
    gb_ch[4,2] = - gb_yz
    gb_ch[5,1] = gb_yz
    
    return gb_ch


def polylog_2(z):
    """
    Evaluation of the polylogarithm functions of order 2 at the unit circle.

    :param z: Argument. Complex number of modulo 1 (np.abs(z) = 1).
    :type z: numpy.complex

    :return: Polylogarithm functions of order 2 at z.
    """
    theta = np.arctan2(z.imag,z.real)

    li2_i = clausen2(theta)

    if theta<0:
        theta = theta + 2*np.pi

    li2_r = np.pi**2/6 - np.pi*theta/2 + theta**2/4

    return li2_r + 1j*li2_i

def polylog_3(z, n_sum = 50):
    """
    Evaluation of the polylogarithm functions of order 3 at the unit circle.

    :param z: Argument. Complex number of modulo 1 (np.abs(z) = 1).
    :type z: numpy.complex

    :return: Polylogarithm functions of order 3 at z.
    """
    n = np.arange(1,n_sum+1)
    li3 = z**n/n**3

    return li3.sum()

def clausen2(x):
    """
    Evaluation of Clausen function of order 2 needed for the evaluation of "polylog_2".

    :param x: The input value.
    :return: The value of the Clausen function of order 2.

    :math:`\mathrm{Cl}_2(x) = - \int_0^x \log \Big[ 2\sin  t/2 \Big]\mathrm{d}t`.

    Original code in Julia by Alexander Voigt with MIT License.
    Python implementation by Diego Romero Abujetas.
    """
    # reduce range of x to [0,pi] for even Clausen functions
    sgn = 1
    if x<0:
        x = -x
        sgn = -1

    if x >= 2*np.pi:
        x = np.mod(x,2*np.pi)

    if x > np.pi:
        x = 2*np.pi - x + 0.0019353071795864769253
        sgn = -sgn

    # evaluation of Clausen2 function
    if x == 0:
        value = 0
    elif x < np.pi/2:
        P = (1.3888888888888889e-02, -4.3286930203743071e-04,
             3.2779814789973427e-06, -3.6001540369575084e-09)
        Q = (1.0000000000000000e+00, -3.6166589746694121e-02,
             3.6015827281202639e-04, -8.3646182842184428e-07)
        y = x*x
        y2 = y*y
        p = P[0] + y * P[1] + y2 * (P[2] + y * P[3])
        q = Q[0] + y * Q[1] + y2 * (Q[2] + y * Q[3])
        value = sgn*x*(1.0 - np.log(x) + y*p/q)
    elif x < np.pi:
        P = (6.4005702446195512e-01, -2.0641655351338783e-01,
             2.4175305223497718e-02, -1.2355955287855728e-03,
             2.5649833551291124e-05, -1.4783829128773320e-07)
        Q = (1.0000000000000000e+00, -2.5299102015666356e-01,
             2.2148751048467057e-02, -7.8183920462457496e-04,
             9.5432542196310670e-06, -1.8184302880448247e-08)
        y = np.pi - x
        z = y*y - np.pi*np.pi/8
        z2 = z*z
        z4 = z2*z2
        p = P[0] + z * P[1] + z2 * (P[2] + z * P[3]) + z4 * (P[4] + z * P[5])
        q = Q[0] + z * Q[1] + z2 * (Q[2] + z * Q[3]) + z4 * (Q[4] + z * Q[5])
        value = sgn*y*p/q

    return value


