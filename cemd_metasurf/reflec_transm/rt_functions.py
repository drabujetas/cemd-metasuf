# Author: Diego Romero Abujetas, March 2023, diegoromeabu@gmail.com
#
# This file contain all the functions needed to calculate reflectance and transmittance.
# of a free standing metasurface of one particle per unit cell. The list of functions included in this file are:
#
# Calculation of reflectance and transmittance.
# - calc_rt_complex, calc_rt, calc_RT_pol

import numpy as np

# calc_rt: Function that calculates complex the specular reflection and transmission (rt) for TE and TM incidence.
#
# Inputs:
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while "th = pi/3" and "a = b" is a triangular lattice.
#
# "gb" is the depolarization Green function, a "6 by 6" matrix.
#
# "alp" is a "6 by 6" matrix with the electric and magnetic polarizability of the particles.
#
# "k" is a scalar with the values of the wavevector in the external media ("k = k0*n_b").
#
# "kx" and "ky" are the projection of the wavevector along "x" and "y" axis.
#
# Outputs:
#
# The output are the complex reflection and transmission (scalars) for different incident
# polarizations (r_tm, t_tm, r_te, t_te).
#

def calc_rt_complex(my_metasurface, alp):
    
    a, b, th = my_metasurface.get_lattice()
    gb = my_metasurface.gb_kxky
    k, kx, ky = my_metasurface.get_bloch()

    alp = k ** 2 * alp   
    gb_alp = np.linalg.inv( np.eye(6) - np.dot(gb, alp) )

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

    return r_tm, r_te, t_tm, t_te 
    
    
# calc_RT: Function that calculates the reflectance and transmittance (RT) for TE and TM incidence.
#
# Inputs:
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while "th = pi/3" and "a = b" is a triangular lattice.
#
# "gb" is the depolarization Green function, a "6 by 6" matrix.
#
# "alp" is a "6 by 6" matrix with the electric and magnetic polarizability of the particles.
#
# "k" is a scalar with the values of the wavevector in the external media ("k = k0*n_b").
#
# "kx0" and "ky0" are the proyection of the wavevector along "x" and "y" axis.
#
# "n" and "m" is the calculated diffraction order ("n = m = 0" is the specular mode).
#
# Outputs:
#
# The output are the reflectance and transmittance (scalars) for different incident
# polarizations (r_tm, t_tm, r_te, t_te).

def calc_rt(my_metasurface, n = 0, m = 0):
    
    a, b, th = my_metasurface.get_lattice()
    gb = my_metasurface.gb_kxky
    alp = my_metasurface.alp_uc
    k, kx0, ky0 = my_metasurface.get_bloch()
    alp = k ** 2 * alp
    
    gb_alp = np.linalg.inv( np.eye(6) - np.dot(gb, alp) )
    
    if k ** 2 - kx0 ** 2 - ky0 ** 2 < 0:
        raise ValueError("The oncoming wave is an evanescent wave")

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
    
    
# calc_RT: Function that calculates the reflectance and transmittance (RT) for an arbitrary plane wave incidence.
#
# Inputs:
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while "th = pi/3" and "a = b" is a triangular lattice.
#
# "gb" is the depolarization Green function, a "6 by 6" matrix.
#
# "alp" is a "6 by 6" matrix with the electric and magnetic polarizability of the particles.
#
# "k" is a scalar with the values of the wavevector in the external media ("k = k0*n_b").
#
# "kx0" and "ky0" are the projection of the wavevector along "x" and "y" axis.
#
# "polTM" and "polTE" determine the projection of the incoming wave into "TM" and "TE" incidence.
# For exaple:
#	- "polTM = 1", "polTE = 0" --> TM incidence
#	- "polTM = 1", "polTE = 1j" --> circular polarized light
#
# "n" and "m" is the calculated diffraction order ("n = m = 0" is the specular mode).
#
# Outputs:
#
# The output are the reflectance and transmittance (scalars) at the specific polarization.
 
def calc_rt_pol(my_metasurface, alp, polTM = 1, polTE = 1j, n = 0, m = 0):
        
    a, b, th = my_metasurface.get_lattice()
    k, kx0, ky0 = my_metasurface.get_bloch()
    alp = k ** 2 * alp
    
    gb_alp = np.linalg.inv( np.eye(6) - np.dot(gb, alp) )
    
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
        Ei = (ei_tm*polTM + ei_te*polTE)/np.abs(polTM + polTE)
        
        if m == 0 and n == 0:
            w_t = 1
        else:
            w_t = 0
        
        Ef = np.dot(gb_alp, Ei)
        
        Er = grf @ alp @ Ef
        Et = gtf @ alp @ Ef + Ei*w_t
        
        EEi = np.transpose(np.conj(Ei)) @ sz @ Ei 

        R = - (np.transpose(np.conj(Er)) @ sz @ Er)/EEi
        T = + (np.transpose(np.conj(Et)) @ sz @ Et)/EEi
	                
    else:
        
        R = 0 
        T = 0 

    return np.real(R), np.real(T) 



 
    
