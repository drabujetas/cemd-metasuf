"""
The file field_functions.py contain all the functions needed to calculate the near fields for the class NearField.

List of functions: 
    - calc_near_field_kxky     (near field)
"""

import numpy as np

def calc_near_field_kxky(self,my_bloch,pol_tm, pol_te, x,y,z):
    """
    Function that calculates the scatterd field by the metasurface when it is illiminated by an external plane wave
    which polarization is characterized by pol_tm and pol_te.

    :param my_bloch: The object with the information of the incomming wavevector. Only the last entry is used.  
    :type my_bloch: classes.BlochWavevector
    :param pol_tm: TM amplitude.
    :type pol_tm: complex
    :param pol_te: TE amplitude.
    :type pol_te: complex
    :param x: Position in the x-axis where the field is calcualted.
    :type x: float 
    :param y: Position in the y-axis  where the field is calcualted.
    :type y: float 
    :param z: Position in the z-axis where the field is calcualted.
    :type z: float

    :return: tuple with the value of the electric and magnetic field.
    """
    a, b, th = self.get_lattice()
    x_uc, y_uc, z_uc = self.get_unit_cell()
    k, kx, ky = my_bloch.get_bloch_i(-1)
    alp = self.alp_uc

    if k ** 2 - kx ** 2 - ky ** 2 < 0:
        raise ValueError("The incoming wave is an evanescent wave")
    
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2)
    ang1 = np.arccos(kz/k)
    alpha2 = np.arctan2(ky,kx)

    eh0_tm = np.transpose(np.array([[np.cos(ang1) * np.cos(alpha2), np.cos(ang1) * np.sin(alpha2), -np.sin(ang1),
                                   -np.sin(alpha2), np.cos(alpha2), 0]]))
    eh0_te = np.transpose(np.array([[np.sin(alpha2), -np.cos(alpha2), 0, np.cos(ang1) * np.cos(alpha2),
                                   np.cos(ang1) * np.sin(alpha2), -np.sin(ang1)]]))
    
    eh0 = (eh0_tm * pol_tm + eh0_te * pol_te) 

    if type(x) == np.float64:
        gf = calc_gf(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc)
        eh_field = gf@alp@eh0
    else:
        return "case not implemented"

    e_field = eh_field[0:3]
    h_field = eh_field[3:6]

    return e_field, h_field

def calc_gf(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc,n_sum=100):

    """
    Function for calculating the Green function of the metasurface (field propagator from the metasurface).

    :param a: Length lattice vector along the x-axis
    :type a: float
    :param b: Length lattice vector along the other axis (defined by the :param th)
    :type b: float
    :param th: Angle between lattice vectors.
    :type th: float
    :param k: Wavevector in the medium.
    :type k: float or numpy.ndarray
    :param kx: Bloch wavevector (Floquet periodicity) along the x-axis.
    :type kx: float or numpy.ndarray
    :param ky: Bloch wavevector (Floquet periodicity) along the y-axis.
    :type ky: float or numpy.ndarray
    :param x: Position in the x-axis where the field is calcualted.
    :type x: float 
    :param y: Position in the y-axis  where the field is calcualted.
    :type y: float 
    :param z: Position in the z-axis where the field is calcualted.
    :type z: float
    :param x_uc: Position in the x-axis from where the field is propagated.
	:type x_uc: float
	:param y_uc: Position in the y-axis from where the field is propagated.
	:type y_uc: float 
	:param z_uc: Position in the z-axis from where the field is propagated.
	:type z_uc: float 
    :param n_sum: Number of elements taken in the sum.
    :type n_sum: int
    
    :return:
    """

    n_l = int(np.floor( np.real(k + np.abs(kx))/(2*np.pi/a) ) + 3) # convergence parameter
    if n_l > 7:
        n_l = 7
        raise ValueError("a/lambda >> 1")

    gf = calc_gf_1puc(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc,n_sum)

    return gf

# These functions calculate the Green function for the propagation of the field from "r = (x, y, z)"
# to "r0 =(x0, y0, z0)" of a wave generated in an 2D periodic array.
# (Be careful with this definition because the field propagates toward "r0", while the field is
# generated at "r". For historical reasons I took the opposite convention).
#
# "N" and "Ni" are the number of elements taken in the sums. For biggre "N"
# the convergenve is better, but be careful with "Ni". With "Ni = 5" is
# enough to get a good convergence, but for bigger "Ni" is necesary to take
# bigger "N" to get convergence. See PRB 2020, D. R. Abuejetas et. al., 102, 125411 
# for more information about the convergence and the menaing of "Ni".
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while for "th = pi/3" and "a = b" a triangular lattice is recovered.
# Due to the inner functions it is
# possible that the code does not work for "th" different than "pi/2". In
# this case, the function "GbCalc_NUC" can be used (where the sums are done
# in real space).
#
# "X", and "Y" are vectors with the possitions of the particles in the unit
# cell. The position of the particle "i" is (X(i),Y(i)). For one particle
# per unit cell set "X = 0" and "Y = 0".
#
# "k" is the wavector in the metasurface medium ("k = k0*n_bg").
#
# "kx" and "ky" are the proyection of the wavevector along "x" and "y" axis.
# They are real quantities, while "k" can be complex.
#
# "x0", "y0" and "z0" are the position of observation of the field (toward where the field propagates).
# "x", "y" and "z" are the position where the field is generated.
# (Note that here the convention is the opposite, the field comes from "r" and goes to "r0").
#
# The first "if" determines is the calculation is done with the cylinder orientated along witch axis. 
# The inner functions was thiking considering that the cylinders axis is along the "x" axis. Then,  
# if the particles are place at "y - y0 = 0" and "z - z0 = 0", these functions does not work (the field is 
# calculated at "rho = 0" and the result would be infinite). For these reason, I use the same functions (for 
# cylinders along the "x" axis) but changing "x -> y" and "y -> - x". Also, this is the reason why they are 
# only valid for "th = pi/2". The change is only tested at "th = pi/2". This means that, if the particles are 
# in "x - x0 = 0" and "z - z0 = 0", the functions work for any "th", but for particles in "y - y0 = 0" 
# and "z - z0 = 0", is not tested.
#
# Outputs:
#
# Components of the Green function.

def calc_gf_1puc(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc,n_sum):

    n_l = int(np.floor( np.real(k + np.abs(kx))/(2*np.pi/a) ) + 3) # convergence parameter
    if n_l > 7:
        n_l = 7
        raise ValueError("a/lambda >> 1")
    
    sxx = 0
    syy = 0
    szz = 0
    sxy = 0
    syz = 0
    szx = 0
    sxyem = 0
    syzem = 0
    szxem = 0
    
    if (y - y_uc) == 0 and (z - z_uc) == 0: #  I only test the "th = pi/2" case
        
        kcy = kx*np.cos(th) + ky*np.sin(th)
        kpcy = - kx*np.sin(th) + ky*np.cos(th)
    
        for i in range(n_l*2 + 1):
            kcyl = kcy - 2*np.pi/b*(i - n_l)
            sxx = sxx + Gyy1D_kx(n_sum,a*np.sin(th),k,kpcy + (kcyl-kcy)*(np.cos(th))/np.sin(th),kcyl,y,-x,z,y_uc,-x_uc,z_uc)
            syy = syy + Gxx1D_kx(n_sum,a*np.sin(th),k,kpcy + (kcyl-kcy)*(np.cos(th))/np.sin(th),kcyl,y,-x,z,y_uc,-x_uc,z_uc)
            szz = szz + Gzz1D_kx(n_sum,a*np.sin(th),k,kpcy + (kcyl-kcy)*(np.cos(th))/np.sin(th),kcyl,y,-x,z,y_uc,-x_uc,z_uc)
            sxy = sxy + Gxy1D_kx(n_sum,a*np.sin(th),k,kpcy + (kcyl-kcy)*(np.cos(th))/np.sin(th),kcyl,y,-x,z,y_uc,-x_uc,z_uc)
            syzem = syzem + GzxEM1D_kx(n_sum,a*np.sin(th),k,kpcy + (kcyl-kcy)*(np.cos(th))/np.sin(th),kcyl,y,-x,z,y_uc,-x_uc,z_uc)
            szxem = szxem + GyzEM1D_kx(n_sum,a*np.sin(th),k,kpcy + (kcyl-kcy)*(np.cos(th))/np.sin(th),kcyl,y,-x,z,y_uc,-x_uc,z_uc)

        sxx = sxx/b
        syy = syy/b
        szz = szz/b
        sxy = sxy/b
        sxyem = sxyem/b
        syzem = syzem/b
        szxem = szxem/b

    else:                                 # valid for any "th"
        
        for i in range(n_l*2 + 1):
            kxl = kx - 2*np.pi/a*(i - n_l)
            sxx = sxx + Gxx1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)
            syy = syy + Gyy1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)
            szz = szz + Gzz1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)
            sxy = sxy + Gxy1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)
            syzem = syzem + GyzEM1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)
            szxem = szxem + GzxEM1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)

        sxx = sxx/a
        syy = syy/a
        szz = szz/a
        sxy = sxy/a
        sxyem = sxyem/a
        syzem = syzem/a
        szxem = szxem/a
    
    if (z - z_uc) != 0:
        for i in range(n_l*2 + 1):
            kxl = kx - 2*np.pi/a*(i - n_l)
            syz = syz + Gyz1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)
            szx = szx + Gzx1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)
            sxyem = sxyem + GxyEM1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)

        syz = syz/a
        szx = szx/a
        sxyem = sxyem/a

    gf = np.zeros((6, 6) , dtype = 'complex_')
    ggs_ee = np.zeros((3, 3) , dtype = 'complex_')
    ggs_me = np.zeros((3, 3) , dtype = 'complex_')
    
    ggs_ee[0,0] = sxx
    ggs_ee[1,1] = syy
    ggs_ee[2,2] = szz
    ggs_ee[0,1] = sxy
    ggs_ee[1,0] = sxy
    ggs_ee[1,2] = syz
    ggs_ee[2,1] = syz
    ggs_ee[2,0] = szx
    ggs_ee[0,2] = szx

    ggs_me[0,1] = -sxyem
    ggs_me[1,0] = sxyem
    ggs_me[1,2] = -syzem
    ggs_me[2,1] = syzem
    ggs_me[2,0] = -szxem
    ggs_me[0,2] = szxem

    gf[0:3,0:3] = ggs_ee
    gf[3:6,3:6] = ggs_ee
    gf[0:3,3:6] = ggs_me
    gf[3:6,0:3] = -ggs_me

    return gf

    
# Functions that calculate the Green function for the propagation of the field from "r = (x, y, z)"
# to "r0 =(x0, y0, z0)" of a wave generated in an periodic array of cylinders with their axis
# along the "x" axis.
# (Be careful with this definition because the field propagates toward "r0", while the field is
# generated at "r". For historical reasons I took the opposite convention).
# Due to the configuration of the lattice, this functions diverges at "y0 - y = 0" and "z0 - z = 0".
#
# Inputs:
#
# "N" is the number of elements considered in the sum.
#
# "a" is the lattice constant of the 1D array (separation between cylinders).
#
# "k" is the wavector in the metasurface medium ("k = k0*n_bg").
#
# "kx" and "ky" are the proyection of the wavevector along "x" and "y" axis.
# They are real quantities, while "k" can be complex.
#
# "x0", "y0" and "z0" are the position of observation of the field (toward where the field propagates).
# "x", "y" and "z" are the position where the field is generated.
# (Note that here the convention is the opposite, the field comes from "r" and goes to "r0").
#
# Outputs:
#
# Green functions terms for 1D arrays

def Gxx1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z) :
    
    m = np.linspace(1,N,N)

    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')
     
    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)
    
    Gxxs = (1j/(2*a)*kp ** 2/k ** 2*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*(1/kz*np.exp(1j*kz*Z) 
        + np.sum( np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)/kzm + np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)/kzmm + 2j/km*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) ) 
        + 1j*a/(2*np.pi)*(np.exp(ky*Z)*np.log( (1 - np.exp(-2*np.pi/a*(Z + 1j*Y)))) + np.exp(-ky*Z)*np.log(1 - np.exp(-2*np.pi/a*(Z - 1j*Y)))) ) )
    
    return Gxxs
    
    
def Gyy1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z):
    
    m = np.linspace(1,N,N)
    
    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')

    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)

    k0 = 2*np.pi/a
    ek0p = np.exp(-k0*(Z + 1j*Y))
    ek0m = np.exp(-k0*(Z - 1j*Y))
    ekyp = np.exp(ky*Z)
    ekym = np.exp(-ky*Z)
    
    Gyys = 1j/(2*a*k**2)*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*((k**2 - ky**2)/kz*np.exp(1j*kz*Z) 
        + np.sum( np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)*(k**2 - kym**2)/kzm + np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)*(k**2 - kymm**2)/kzmm 
        + 1j*(k**2 + kx**2 - 2*km**2 - Z*kp**2*km)/km*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) + 2*ky*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z)) 
        + 1j*k0*(ekyp*ek0p/(1 - ek0p) ** 2 + ekym*ek0m/(1 - ek0m) ** 2) 
        - 1j*ky*(ekyp*ek0p/(1-ek0p) - ekym*ek0m/(1-ek0m))
        + 1j/2*Z*kp**2*(ekyp*ek0p/(1-ek0p) + ekym*ek0m/(1-ek0m))
        + 1j/2*(k**2 + kx**2)/k0*(ekyp*np.log(1 - ek0p) + ekym*np.log(1 - ek0m) ) )
    
    return Gyys


def Gzz1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z):
    
    m = np.linspace(1,N,N)
            
    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')

    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)
    
    k0 = 2*np.pi/a
    ek0p = np.exp(-k0*(Z + 1j*Y))
    ek0m = np.exp(-k0*(Z - 1j*Y))
    ekyp = np.exp(ky*Z)
    ekym = np.exp(-ky*Z)
    
    Gzzs = 1j/(2*a*k**2)*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*((k**2 - kz**2)/kz*np.exp(1j*kz*Z) 
        + np.sum( np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)*(k**2 - kzm**2)/kzm + np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)*(k**2 - kzmm**2)/kzmm 
        + 1j*(k**2 + kx**2 + 2*km**2 + Z*kp**2*km + 1/4*Z**2*kp**4)/km*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) - 2*ky*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z)) 
        - 1j*k0*(ekyp*ek0p/(1 - ek0p) ** 2 + ekym*ek0m/(1 - ek0m) ** 2) 
        + 1j*ky*(ekyp*ek0p/(1-ek0p) - ekym*ek0m/(1-ek0m))
        - 1j/2*Z*kp**2*(ekyp*ek0p/(1-ek0p) + ekym*ek0m/(1-ek0m))
        + 1j/2*(k**2 + kx**2 + 1/4*Z**2*kp**4)/k0*(ekyp*np.log(1 - ek0p) + ekym*np.log(1 - ek0m)) )
    
    return Gzzs


def Gxy1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z):
    
    m = np.linspace(1,N,N)
            
    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')
           
    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)
     
    Gxys = ( 1j/(2*a*k ** 2)*kx*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*( - ky/kz*np.exp(1j*kz*Z) 
        + np.sum( - np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)*kym/kzm - np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)*kymm/kzmm
        + 2*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z)) # I think that 1/km -> log() are mising (looks unecessary for convergence).
        - 1j*(np.exp(ky*Z)*np.exp(-2*np.pi/a*(Z + 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z + 1j*Y))) - np.exp(-ky*Z)*np.exp(-2*np.pi/a*(Z - 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z - 1j*Y))) ) ) )
    
    return Gxys


def GzxEM1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z):
    
    m = np.linspace(1,N,N)

    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')
           
    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)
    
    GzxEMs = (1j/(2*a*k)*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*( - ky/kz*np.exp(1j*kz*Z) 
        + np.sum( - np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)*kym/kzm - np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)*kymm/kzmm 
        + 2*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z)) # I think that 1/km -> log() are mising (same as Gbxy).
        - 1j*(np.exp(ky*Z)*np.exp(-2*np.pi/a*(Z + 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z + 1j*Y))) - np.exp(-ky*Z)*np.exp(-2*np.pi/a*(Z - 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z - 1j*Y))) ) ) )
    
    return GzxEMs
    

def GyzEM1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z):
    
    m = np.linspace(1,N,N)
            
    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')
           
    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)
    
    GyzEMs = (-1j/(2*a)*kx/k*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*(1/kz*np.exp(1j*kz*Z) 
        + np.sum( np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)/kzm + np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)/kzmm + 2j/km*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) ) 
        + 1j*a/(2*np.pi)*(np.exp(ky*Z)*np.log( (1 - np.exp(-2*np.pi/a*(Z + 1j*Y)))) + np.exp(-ky*Z)*np.log(1 - np.exp(-2*np.pi/a*(Z - 1j*Y)))) ) )
    
    return GyzEMs


# The next function are zero when "z = z0" and change the sign when "z <--> z0"
  
    
def Gyz1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z):
    
    m = np.linspace(1,N,N)
            
    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')
    
    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)
    
    if Z == 0:
        Gyzs = 0
    else:
        Gyzs = ( 1j/(2*a*k ** 2)*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*( - ky*np.exp(1j*kz*Z) 
            + np.sum( - np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)*kym - np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)*kymm 
            + 2*ky*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) + 2j*km*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z))
            - 1*ky*(np.exp(ky*Z)*np.exp(-2*np.pi/a*(Z + 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z + 1j*Y))) + np.exp(-ky*Z)*np.exp(-2*np.pi/a*(Z - 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z - 1j*Y))) ) 
            + 1*2*np.pi/a*(  np.exp(ky*Z)*np.exp(2*np.pi/a*(Z + 1j*Y))/(1 - np.exp(2*np.pi/a*(Z + 1j*Y))) ** 2 -  np.exp(-ky*Z)*np.exp(2*np.pi/a*(Z - 1j*Y))/(1 - np.exp(2*np.pi/a*(Z - 1j*Y))) ** 2 )) )
    
    Gyzs = Gyzs*(z0 - z)/np.abs(z0 - z)
    
    return Gyzs


def Gzx1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z):
    
    m = np.linspace(1,N,N)
            
    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')

    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)
    
    if Z == 0:
        Gzxs = 0
    else:
        Gzxs = (1j/(2*a*k ** 2)*kx*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*( - np.exp(1j*kz*Z) 
            + np.sum( - np.exp(1j*kzm*Z)*np.exp(-1j*km*Y) - np.exp(1j*kzmm*Z)*np.exp(1j*km*Y) 
            + 2*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z))
            - 1*(np.exp(ky*Z)*np.exp(-2*np.pi/a*(Z + 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z + 1j*Y))) + np.exp(-ky*Z)*np.exp(-2*np.pi/a*(Z - 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z - 1j*Y))) ) ) )

    Gzxs = Gzxs*(z0 - z)/np.abs(z0 - z)

    return Gzxs


def GxyEM1D_kx(N,a,k,ky,kx,x0,y0,z0,x,y,z): 
    
    m = np.linspace(1,N,N)

    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')

    km = 2*np.pi*m/a
    kym = ky - km
    kymm = ky + km
    kzm = np.sqrt(k ** 2 - kx ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(k ** 2 - kx ** 2 - kymm ** 2, dtype = 'complex_')

    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)
    
    if Z == 0:
        GxyEMs = 0
    else:
        GxyEMs = ( 1j/(2*a*k)*np.exp(1j*ky*Y)*np.exp(1j*kx*X)*( - np.exp(1j*kz*Z) 
            + np.sum( - np.exp(1j*kzm*Z)*np.exp(-1j*km*Y) - np.exp(1j*kzmm*Z)*np.exp(1j*km*Y) 
            + 2*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z))
            - 1*(np.exp(ky*Z)*np.exp(-2*np.pi/a*(Z + 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z + 1j*Y))) + np.exp(-ky*Z)*np.exp(-2*np.pi/a*(Z - 1j*Y))/(1 - np.exp(-2*np.pi/a*(Z - 1j*Y))) ) ) )
        
    GxyEMs = GxyEMs*(z0 - z)/np.abs(z0 - z)
    
    return GxyEMs
        