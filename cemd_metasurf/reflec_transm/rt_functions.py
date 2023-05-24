"""
The file rt_functions.py contain all the functions needed to calculate the reflection and tranmision.

List of functions: 
    - calc_rt_complex   (reflectivity and transmitivity for TE and TM waves)
    - calc_rt          (reflectance and transmitance for TE and TM waves)
    - calc_rt_pol      (reflectance and transmitance for a given polarization)
"""

import numpy as np

def calc_rt_complex(my_metasurface):
    """
    Function that calculates complex amplitude specular reflection and transmission (r and t) for TE and TM incidence waves.

    :param my_metasurface: Vacuum wavevector.
    :type my_metasurface: classes.Metasurface

    :return: array with complex amplitude specular reflection and transmissionat both polarizations
    """
    a, b, th = my_metasurface.get_lattice()
    gb = my_metasurface.gb_kxky
    alp = my_metasurface.alp_uc
    k, kx, ky = my_metasurface.get_bloch()

    alp = k ** 2 * alp   
    gb_alp = np.linalg.inv( np.eye(6) - np.dot(gb, alp) )

    if k ** 2 - kx ** 2 - ky ** 2 < 0:
        raise ValueError("The incoming wave is an evanescent wave")

    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2)
    ang1 = np.arccos(kz/k)
    alpha2 = np.arctan2(ky,kx)
        
    gfeer = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,kz*ky/k ** 2],[kx*kz/k ** 2,ky*kz/k ** 2,1-kz ** 2/k ** 2]])
    gfmer = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[0,kz/k,ky/k],[-kz/k,0,-kx/k],[-ky/k,kx/k,0]])
    grf = np.zeros((6,6), dtype = 'complex_')
    grf[0:3,0:3] = gfeer
    grf[0:3,3:6] = -gfmer
    grf[3:6,0:3] = gfmer
    grf[3:6,3:6] = gfeer

    gfeet = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,-kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,-kz*ky/k ** 2],[-kx*kz/k ** 2,-ky*kz/k ** 2,1-kz ** 2/k ** 2]])
    gfmet = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[0,-kz/k,ky/k],[kz/k,0,-kx/k],[-ky/k,kx/k,0]])        
    gtf = np.zeros((6,6), dtype = 'complex_')
    gtf[0:3,0:3] = gfeet
    gtf[0:3,3:6] = -gfmet
    gtf[3:6,0:3] = gfmet
    gtf[3:6,3:6] = gfeet

    ei_tm = np.transpose(np.array([[np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1),-np.sin(alpha2),np.cos(alpha2),0]]))
    ei_te = np.transpose(np.array([[np.sin(alpha2),-np.cos(alpha2),0,np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1)]]))

    ef_tm = np.dot(gb_alp, ei_tm)
    ef_te = np.dot(gb_alp, ei_te)

    er_tm = grf @ alp @ ef_tm
    er_te = grf @ alp @ ef_te
    et_tm = (gtf @ alp @ ef_tm) + ei_tm
    et_te = (gtf @ alp @ ef_te) + ei_te

    if np.abs(ei_tm[4,0]) > np.abs(ei_tm[3,0]):
        r_tm = er_tm[4,0]/ei_tm[4,0] 
        t_tm = et_tm[4,0]/ei_tm[4,0]
        r_te = er_te[1,0]/ei_te[1,0]
        t_te = et_te[1,0]/ei_te[1,0]
    else:
        r_tm = er_tm[3,0]/ei_tm[3,0] 
        t_tm = et_tm[3,0]/ei_tm[3,0]
        r_te = er_te[0,0]/ei_te[0,0]
        t_te = et_te[0,0]/ei_te[0,0]

    return np.array([r_tm,r_te,t_tm,t_te]) 


def calc_rt(my_metasurface, n = 0, m = 0):
    """
    Function that calculates reflectance and transmitance (R and T) for TE and TM incidence waves at different
    diffractive orders.

    :param my_metasurface: Vacuum wavevector.
    :type my_metasurface: classes.Metasurface
    :param n: Diffractive order along x-axis.
    :type n: int
    :param m: Diffractive order along the other axis (y-axis for rectangular arrays).
    :type m: int

    :return: array with the reflectance and transmitance at both polarizations.
    """
    a, b, th = my_metasurface.get_lattice()
    gb = my_metasurface.gb_kxky
    alp = my_metasurface.alp_uc
    k, kx0, ky0 = my_metasurface.get_bloch()
    alp = k ** 2 * alp
    
    gb_alp = np.linalg.inv( np.eye(6) - np.dot(gb, alp) )
    
    if k ** 2 - kx0 ** 2 - ky0 ** 2 < 0:
        raise ValueError("The incoming wave is an evanescent wave")

    kz0 = np.sqrt(k ** 2 - kx0 ** 2 - ky0 ** 2) #by definition, must be real
    ang1 = np.arccos(kz0/k)
    alpha2 = np.arctan2(ky0,kx0)
    
    kx = kx0 - 2*np.pi/a*n
    ky = ky0 + 2*np.pi*n/a*np.sin(np.pi/2 - th)/np.cos(np.pi/2 - th) - 2*np.pi*m/(b*np.cos(np.pi/2 - th))
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')
    
    if np.imag(kz) == 0:
    
        kz = kz.real

        gfeer = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,kz*ky/k ** 2],[kx*kz/k ** 2,ky*kz/k ** 2,1-kz ** 2/k ** 2]])
        gfmer = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[0,kz/k,ky/k],[-kz/k,0,-kx/k],[-ky/k,kx/k,0]])
        grf = np.zeros((6,6), dtype = 'complex_')
        grf[0:3,0:3] = gfeer
        grf[0:3,3:6] = -gfmer
        grf[3:6,0:3] = gfmer
        grf[3:6,3:6] = gfeer

        gfeet = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,-kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,-kz*ky/k ** 2],[-kx*kz/k ** 2,-ky*kz/k ** 2,1-kz ** 2/k ** 2]])
        gfmet = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[0,-kz/k,ky/k],[kz/k,0,-kx/k],[-ky/k,kx/k,0]])        
        gtf = np.zeros((6,6), dtype = 'complex_')
        gtf[0:3,0:3] = gfeet
        gtf[0:3,3:6] = -gfmet
        gtf[3:6,0:3] = gfmet
        gtf[3:6,3:6] = gfeet
        
        #sz = 1/(4)*np.array([[0,0,0,0,1,0],[0,0,0,-1,0,0],[0,0,0,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]])
        
        ei_tm = np.transpose(np.array([[np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1),-np.sin(alpha2),np.cos(alpha2),0]]))
        ei_te = np.transpose(np.array([[np.sin(alpha2),-np.cos(alpha2),0,np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1)]]))
        
        if m == 0 and n == 0:
            w_t = 1
        else:
            w_t = 0
        
        ef_tm = np.dot(gb_alp, ei_tm)
        ef_te = np.dot(gb_alp, ei_te)
        
        er_tm = grf @ alp @ ef_tm
        et_tm = gtf @ alp @ ef_tm + ei_tm*w_t
        er_te = grf @ alp @ ef_te
        et_te = gtf @ alp @ ef_te + ei_te*w_t
        
        eei_tm = np.cos(ang1)#np.transpose(np.conj(ei_tm)) @ sz @ ei_tm 
        eei_te = np.cos(ang1)#np.transpose(np.conj(ei_te)) @ sz @ ei_te 

        r_tm = - (np.conj(er_tm[4,0])*er_tm[0,0] - np.conj(er_tm[3,0])*er_tm[1,0]).real/eei_tm
        t_tm = + (np.conj(et_tm[4,0])*et_tm[0,0] - np.conj(et_tm[3,0])*et_tm[1,0]).real/eei_tm
        r_te = - (np.conj(er_te[4,0])*er_te[0,0] - np.conj(er_te[3,0])*er_te[1,0]).real/eei_te
        t_te = + (np.conj(et_te[4,0])*et_te[0,0] - np.conj(et_te[3,0])*et_te[1,0]).real/eei_te
        
    else:
        
        t_tm = 0 
        r_tm = 0
        t_te = 0
        r_te = 0

    return np.array([r_tm,r_te,t_tm,t_te])
    

def calc_rt_pol(my_metasurface, pol_tm = 1, pol_te = 1j, n = 0, m = 0):
    """
    Function that calculates reflectance and transmitance (R and T) for a given incidence wave at different
    diffractive orders. By default is a circular polarized wave.

    :param my_metasurface: Vacuum wavevector.
    :type my_metasurface: classes.Metasurface
    :param pol_tm: TM amplitude.
    :type pol_tm: complex
    :param pol_te: TE amplitude.
    :type pol_te: complex
    :param n: Diffractive order along x-axis.
    :type n: int
    :param m: Diffractive order along the other axis (y-axis for rectangular arrays).
    :type m: int

    :return: array with the reflectance and transmitance.
    """
    a, b, th = my_metasurface.get_lattice()
    gb = my_metasurface.gb_kxky
    k, kx0, ky0 = my_metasurface.get_bloch()
    alp = my_metasurface.alp_uc

    alp = k ** 2 * alp
    
    gb_alp = np.linalg.inv( np.eye(6) - np.dot(gb, alp) )
    
    if k ** 2 - kx0 ** 2 - ky0 ** 2 < 0:
        raise ValueError("The incoming wave is an evanescent wave")

    kz0 = np.sqrt(k ** 2 - kx0 ** 2 - ky0 ** 2) #by definition, must be real
    ang1 = np.arccos(kz0/k)
    alpha2 = np.arctan2(ky0,kx0)
    
    kx = kx0 - 2*np.pi/a*n
    ky = ky0 + 2*np.pi*n/a*np.sin(np.pi/2 - th)/np.cos(np.pi/2 - th) - 2*np.pi*m/(b*np.cos(np.pi/2 - th))
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')
    
    if np.imag(kz) == 0:
    
        pre_fac = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)
    
        gfeer = pre_fac*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,kz*ky/k ** 2],[kx*kz/k ** 2,ky*kz/k ** 2,1-kz ** 2/k ** 2]])
        gfmer = pre_fac*np.array([[0,kz/k,ky/k],[-kz/k,0,-kx/k],[-ky/k,kx/k,0]])
        grf = np.zeros((6,6), dtype = 'complex_')
        grf[0:3,0:3] = gfeer
        grf[0:3,3:6] = -gfmer
        grf[3:6,0:3] = gfmer
        grf[3:6,3:6] = gfeer

        gfeet = pre_fac*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,-kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,-kz*ky/k ** 2],[-kx*kz/k ** 2,-ky*kz/k ** 2,1-kz ** 2/k ** 2]])
        gfmet = pre_fac*np.array([[0,-kz/k,ky/k],[kz/k,0,-kx/k],[-ky/k,kx/k,0]])        
        gtf = np.zeros((6,6), dtype = 'complex_')
        gtf[0:3,0:3] = gfeet
        gtf[0:3,3:6] = -gfmet
        gtf[3:6,0:3] = gfmet
        gtf[3:6,3:6] = gfeet
            
        sz = 1/(4)*np.array([[0,0,0,0,1,0],[0,0,0,-1,0,0],[0,0,0,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]])
              
        ei_tm = np.transpose(np.array([[np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1),-np.sin(alpha2),np.cos(alpha2),0]]))
        ei_te = np.transpose(np.array([[np.sin(alpha2),-np.cos(alpha2),0,np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1)]]))
        ei = (ei_tm*pol_tm + ei_te*pol_te)/np.abs(pol_tm + pol_te)
        
        if m == 0 and n == 0:
            w_t = 1
        else:
            w_t = 0
        
        ef = np.dot(gb_alp, ei)
        
        er = grf @ alp @ ef
        et = gtf @ alp @ ef + ei*w_t
        
        eei = np.transpose(np.conj(ei)) @ sz @ ei 

        r_pol = - (np.transpose(np.conj(er)) @ sz @ er)/eei
        t_pol = + (np.transpose(np.conj(et)) @ sz @ et)/eei
	                
    else:
        
        r_pol = 0 
        t_pol = 0 

    return np.array([r_pol.real[0],t_pol.real[0]])



 
    
