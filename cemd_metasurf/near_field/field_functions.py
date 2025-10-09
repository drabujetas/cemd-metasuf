"""
The file field_functions.py contain all the functions needed to calculate the near fields for the class NearField.

List of functions: 
    - calc_near_field_kxky     (near field)
"""

import numpy as np

def calc_near_field_kxky(my_metasurface,pol_tm, pol_te, x,y,z):
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
    a, b, th = my_metasurface.get_lattice()
    x_uc, y_uc, z_uc = my_metasurface.get_unit_cell()
    k, kx, ky = my_metasurface.get_bloch()
    alp = my_metasurface.alp_uc
    gb = my_metasurface.gb_kxky
    alp = k ** 2 * alp
    
    gb_alp = np.linalg.inv( np.eye(6) - np.dot(gb, alp) )

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
    eh = gb_alp @ eh0

    if np.size(x) == 1:
        gf = calc_gf(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc)
        eh_field = gf@alp@eh
    else:
        return "case not implemented"

    e_field = eh_field[0:3]
    h_field = eh_field[3:6]

    return e_field, h_field

def calc_gf(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc,n_sum=10):

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
    
    :return: numpy.ndarray with the lattce Green function at the given position
    """

    n_l = int(np.floor( np.real(k + np.abs(kx))/(2*np.pi/a) ) + 3) 
    if n_l > 7:
        n_l = 7
        raise ValueError("a/lambda >> 1")

    if np.size(x) == 1:
        gf = calc_gf_1puc(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc,n_sum)
    else:
        return print("the case of complex unit cells is not implemented")

    return gf


def G_em_renorm(k,x,y,z,x_uc,y_uc,z_uc):

    """
    Function for calculating the electromagnetic Green function.

    :param k: Wavevector in the medium.
    :type k: float
    :param x: Position in the x-axis where the field is calcualted.
    :type x: float 
    :param y: Position in the y-axis where the field is calcualted.
    :type y: float 
    :param z: Position in the z-axis where the field is calcualted.
    :type z: float
    :param x_uc: Position in the x-axis from where the field is propagated.
	:type x_uc: float
	:param y_uc: Position in the y-axis from where the field is propagated.
	:type y_uc: float 
	:param z_uc: Position in the z-axis from where the field is propagated.
	:type z_uc: float 
    
    :return: numpy.ndarray with the Green function
    """

    kr1 = k * np.array([x, y, z])
    kr2 = k * np.array([x_uc, y_uc, z_uc])
    kR_vec = kr1-kr2
    kR = np.linalg.norm(kR_vec)
    kR2 = kR**2
    Ur = kR_vec/kR

    term1 = np.exp(1j*kR)/(kR)
    term2 = 1+(1j/(kR))-(1/(kR2))
    term3 = 1+(3*1j/(kR))-(3/(kR2))
    term4 = (1j*kR-1)/kR
    matrix = np.outer(kR_vec, kR_vec)/kR2
    id3 = np.eye(3)
    mat = np.array([[0, -Ur[2], Ur[1]],[Ur[2], 0, -Ur[0]],[-Ur[1], Ur[0], 0]]) 

    Ge = term1*(term2*id3-term3*matrix)
    Gm = 1j*term1*term4*mat 

    G_6x6 = np.block([
                [Ge,  Gm],
                [-Gm, Ge]
            ])
    
    return G_6x6


def k0_mine(val):

    """
    Function for calculating the modified bessel function of second kind of order 0. The implementation is not very acurated, but it is simple

    :param val: Values where k0 is calculated
    :type val: float or numpy.ndarray
    
    :return: float or numpy.ndarray
    """

    gE = 0.577215664901532860606512090082402431042
    result = np.empty_like(val)
    mask = val > 0.8
    
    if np.any(mask):
        val_high = val[mask]
        result[mask] = np.sqrt(np.pi/(2*val_high))*np.exp(-val_high)*(1 - 1/(8*val_high) + 9/(2*(8*val_high)**2) - 9*25/(6*(8*val_high)**3) + 1/3*9*25*49/(24*(8*val_high)**4))
    
    if np.any(~mask):
        val_low = val[~mask]
        result[~mask] = -np.log(val_low/2)*(1 + 0/4*val_low**2) - gE + val_low**2/2 - val_low**4/4 + val_low**6/19
    
    return result

def k1_mine(val):

    """
    Function for calculating the modified bessel function of second kind or order 1. The implementation is not very acurated, but it is simple

    :param val: Values where k1 is calculated
    :type val: float or numpy.ndarray
    
    :return: float or numpy.ndarray
    """
        
    gE = 0.577215664901532860606512090082402431042
    result = np.empty_like(val)
    mask = val > .75
    
    if np.any(mask):
        val_high = val[mask]
        val8 = 8*val_high
        mu = 4
        result[mask] = np.sqrt(np.pi/(2*val_high))*np.exp(-val_high)*(1 + (mu-1)/(val8) + (mu-1)*(mu-9)/(2*(val8)**2) 
                            + (mu-1)*(mu-9)*(mu-25)/(6*(val8)**3) + 1/3*(mu-1)*(mu-9)*(mu-25)*(mu-49)/(24*(val8)**4))
    
    if np.any(~mask):
        val_low = val[~mask]
        result[~mask] = 1/val_low + np.log(val_low/2)/2*(val_low - val_low**2*0.06) - (gE/2 - 1/4)*val_low #- val_low*0.55 + val_low**2*0.08
        
    return result


def calc_gf_1puc(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc,n_sum,rec=True):

    """
    Function for calculating the Green function of the metasurface (field propagator from the metasurface) for one particle per unit cell.

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
    :param rec: Trick to avoid recursive calculation in the second if.
    :type rec: bool
    
    :return: numpy.ndarray with the lattce Green function at the given position
    """

    n_l = int(np.floor( np.real(k + np.abs(kx))/(2*np.pi/a) ) + 5) # convergence parameter
    if n_l > 10:
        n_l = 10
        raise ValueError("a/lambda >> 1")
    
    if np.sqrt( (x - x_uc)**2 + (y - y_uc)**2 + (z - z_uc)**2 ) < 0.05*a and rec == True:
        im_p = np.imag(calc_gf_1puc(a,b,th,k,kx,ky,x,y,z,x_uc,y_uc,z_uc,n_sum=10,rec = False))
        re_p = G_em_renorm(k,x,y,z,x_uc,y_uc,z_uc)/(4*np.pi)*k
        return np.real(re_p) + 1j*im_p
    
    gf = np.zeros((6, 6) , dtype = 'complex_')

    if (y - y_uc) == 0 and (z - z_uc) == 0: #  Managing the case along the x-axis 
    # It looks like it is working for any "th"
    # A rotate the MTs an angle "th", then I have a latice with "a <-> b" and "th' = pi - th",
    # and finally I rotate back an angle "-th" the Green function.
        
        cthp = cth = 0
        sthp = sth = 1
        if th != np.pi/2:
            cth = np.cos(th)
            sth = np.sin(th)
            thp = np.pi - th
            cthp = np.cos(thp)
            sthp = np.sin(thp)

        kcy = kx*cth + ky*sth
        kpcy = - kx*sth + ky*cth
        
        xr = x*cth + y*sth
        yr = -x*sth + y*cth
        x_uc_r = x_uc*cth + y_uc*sth
        y_uc_r = -x_uc*sth + y_uc*cth
        
        gf = np.zeros((6, 6) , dtype = 'complex_')

        for i in range(n_l*2 + 1):
            kcyl = kcy - 2*np.pi/b*(i - n_l)
            gf += G1D_kx(n_sum,a*sthp,k,kpcy - (kcyl-kcy)*(cthp)/sthp,kcyl,xr,yr,z,x_uc_r,y_uc_r,z_uc)
            
        rm_rot = np.array([
                [cth, -sth, 0],
                [sth,  cth, 0],
                [0,          0,         1]
        ])

        gf[0:3,0:3] = gf[3:6,3:6] = (rm_rot @ gf[0:3,0:3] @ rm_rot.T)
        gf[3:6,0:3] = (rm_rot @ gf[3:6,0:3] @ rm_rot.T)
        gf[0:3,3:6] = - gf[3:6,0:3]

        gf = gf/b
        
    else:                                 # valid for any "th"
        
        for i in range(n_l*2 + 1):
            kxl = kx - 2*np.pi/a*(i - n_l)
            gf += G1D_kx(n_sum,b*np.sin(th),k,ky - ((kxl-kx)*np.cos(th)/np.sin(th)),kxl,x,y,z,x_uc,y_uc,z_uc)

        gf = gf/a

    return gf

    
def G1D_kx(n_sum,a,k,ky,kx,x0,y0,z0,x,y,z):

    """
    Function for calculating the lattice Green function for arrays of cylinders (1D case)

    :param a: Length lattice vector along the x-axis
    :type a: float
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

    """

    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')
    k0 = 2*np.pi/a

    X = x0 - x
    Y = y0 - y
    Z = np.abs(z0 - z)

    Hmax = n_sum + 1
    if Z != 0:
        Hmax = int(np.floor(-np.log(1e-50)/np.abs(Z)*a/(2*np.pi)/np.sqrt(2)) + 1)

    if Hmax < n_sum:
        m = np.linspace(1,Hmax,Hmax)

        k2 = k ** 2
        kp2 = k2 - kx ** 2 # np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')

        km = k0*m
        kym = ky - km
        kymm = ky + km
        kzm = np.sqrt(kp2 - kym ** 2, dtype = 'complex_')
        kzmm = np.sqrt(kp2 - kymm ** 2, dtype = 'complex_')
        ekzmky = np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)
        ekzmmky = np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)
        phase = np.exp(1j*ky*Y)*np.exp(1j*kx*X)

        txx = 1j/(2*a)*phase*(1/kz*np.exp(1j*kz*Z) 
            + np.sum( ekzmky/kzm + ekzmmky/kzmm ) ) 
        txy = 1j/(2*a*k)*phase*( - ky/kz*np.exp(1j*kz*Z) 
            + np.sum( - ekzmky*kym/kzm - ekzmmky*kymm/kzmm ) ) 

        Gxxs = kp2/k2*txx
        Gyys = 1j/(2*a*k2)*phase*((k2 - ky**2)/kz*np.exp(1j*kz*Z) 
            + np.sum( ekzmky*(k2 - kym**2)/kzm + ekzmmky*(k2 - kymm**2)/kzmm ) )
        Gzzs = 1j/(2*a*k2)*phase*((k2 - kz**2)/kz*np.exp(1j*kz*Z) 
            + np.sum( ekzmky*(k2 - kzm**2)/kzm + ekzmmky*(k2 - kzmm**2)/kzmm  ) )
        Gxys = kx/k*txy
        GzxEMs = txy
        GyzEMs = -kx/k*txx

        if Z == 0:
            Gyzs = 0
            Gzxs = 0
            GxyEMs = 0
        else: 
            tzx =  1j/(2*a*k)*phase*( - np.exp(1j*kz*Z) + np.sum( - ekzmky - ekzmmky) )
            Gyzs = 1j/(2*a*k2)*phase*( - ky*np.exp(1j*kz*Z) + np.sum( - ekzmky*kym - ekzmmky*kymm) ) 
            Gzxs = kx/k*tzx
            GxyEMs = tzx
    else:
        
        kp = np.sqrt(k**2 - kx**2, dtype = 'complex_')
    
        if np.abs(kx) > k and np.abs(kp)*a > 2:
            m = np.linspace(-n_sum,n_sum,2*n_sum+1)
            kp = np.imag(kp)
            rho = np.sqrt((Y - m*a)**2 + Z**2)
            kpma = kp*rho
            k0_kp = k0_mine(kpma)
            k1_kp = k1_mine(kpma)
            txx = np.exp(1j*kx*X)*np.sum(k0_kp*np.exp(1j*ky*m*a)) /(2*np.pi)
            txy = kp/k*np.exp(1j*kx*X)*np.sum(k0_kp*(Y-m*a)/rho*np.exp(1j*ky*m*a)) /(2*np.pi)

            Gxxs = -(kp**2/k**2)*txx
            Gyys = np.exp(1j*kx*X)*np.sum( (k0_kp*(1 + kp**2/k**2*Y**2/rho**2) + k1_kp*kp**2/k**2/kpma*(2*Y**2/rho**2 -1 ))*np.exp(1j*ky*m*a) ) /(2*np.pi)
            Gzzs = np.exp(1j*kx*X)*np.sum( np.exp(1j*ky*m*a)*(k0_kp*(1 + kp**2/k**2*Z**2/rho**2) + k1_kp*kp**2/k**2/kpma*(2*Z**2/rho**2 -1 )) ) /(2*np.pi)
            Gxys = -1j*kx/k*txy
            GzxEMs = -1j*txy
            GyzEMs = -kx/k*txx

            tzx = -1j*np.exp(1j*kx*X)*kp/k*Z*np.sum(k1_kp/rho*np.exp(1j*ky*m*a)) /(2*np.pi)
            Gyzs = np.exp(1j*kx*X)*kp**2/k**2*Z*np.sum( (Y-m*a)/rho**2*(k0_kp + 2*k1_kp/kpma )*np.exp(1j*ky*m*a)) /(2*np.pi)
            Gzxs = kx/k*tzx
            GxyEMs = tzx
        else:
    
            kp2 = k ** 2 - kx ** 2 #np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
            m = np.linspace(1,n_sum,n_sum)
            
            km = k0*m
            kym = ky - km
            kymm = ky + km
            kzm = np.sqrt(kp2 - kym ** 2, dtype = 'complex_')
            kzmm = np.sqrt(kp2 - kymm ** 2, dtype = 'complex_')

            ek0p = np.exp(-k0*(Z + 1j*Y))
            ek0m = np.exp(-k0*(Z - 1j*Y))
            ekyp = np.exp(ky*Z)
            ekym = np.exp(-ky*Z)
            ekzmky = np.exp(1j*kzm*Z)*np.exp(-1j*km*Y)
            ekzmmky = np.exp(1j*kzmm*Z)*np.exp(1j*km*Y)
            phase = np.exp(1j*ky*Y)*np.exp(1j*kx*X)

            txx =  1j/(2*a)*phase*(1/kz*np.exp(1j*kz*Z) 
                + np.sum( ekzmky/kzm + ekzmmky/kzmm + 2j/km*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) ) 
                + 1j/k0*( ekyp*np.log(1 - ek0p) + ekym*np.log(1 - ek0m) ) )
            txy = 1j/(2*a*k)*phase*( - ky/kz*np.exp(1j*kz*Z) 
                + np.sum( - ekzmky*kym/kzm - ekzmmky*kymm/kzmm 
                + 2*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z))
                - 1j*(ekyp*ek0p/(1 - ek0p) - ekym*ek0m/(1 - ek0m) ) ) 
            
            Gxxs = kp2/k ** 2*txx 

            Gyys = 1j/(2*a*k**2)*phase*( (k**2 - ky**2)/kz*np.exp(1j*kz*Z) 
                + np.sum( ekzmky*(k**2 - kym**2)/kzm + ekzmmky*(k**2 - kymm**2)/kzmm 
                + 1j*(k**2 + kx**2 - 2*km**2 - Z*kp**2*km)/km*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) + 2*ky*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z)) 
                + 1j*k0*(ekyp*ek0p/(1 - ek0p) ** 2 + ekym*ek0m/(1 - ek0m) ** 2) 
                - 1j*ky*(ekyp*ek0p/(1 - ek0p) - ekym*ek0m/(1 - ek0m))
                + 1j/2*Z*kp**2*(ekyp*ek0p/(1 - ek0p) + ekym*ek0m/(1 - ek0m))
                + 1j/2*(k**2 + kx**2)/k0*(ekyp*np.log(1 - ek0p) + ekym*np.log(1 - ek0m) ) )
            
            Gzzs = 1j/(2*a*k**2)*phase*((k**2 - kz**2)/kz*np.exp(1j*kz*Z) 
                + np.sum( ekzmky*(k**2 - kzm**2)/kzm + ekzmmky*(k**2 - kzmm**2)/kzmm 
                + 1j*(k**2 + kx**2 + 2*km**2 + Z*kp**2*km + 1/4*Z**2*kp**4)/km*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) - 2*ky*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z)) 
                - 1j*k0*(ekyp*ek0p/(1 - ek0p) ** 2 + ekym*ek0m/(1 - ek0m) ** 2) 
                + 1j*ky*(ekyp*ek0p/(1-ek0p) - ekym*ek0m/(1-ek0m))
                - 1j/2*Z*kp**2*(ekyp*ek0p/(1-ek0p) + ekym*ek0m/(1-ek0m))
                + 1j/2*(k**2 + kx**2 + 1/4*Z**2*kp**4)/k0*(ekyp*np.log(1 - ek0p) + ekym*np.log(1 - ek0m)) )

            Gxys = kx/k*txy
            GzxEMs = txy
            GyzEMs = -kx/k*txx 
            
            if Z == 0:
                Gyzs = 0
                Gzxs = 0
                GxyEMs = 0
            else:
                tzx = 1j/(2*a*k)*phase*( - np.exp(1j*kz*Z) 
                    + np.sum( - ekzmky - ekzmmky 
                    + 2*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z))
                    - 1*(ekyp*ek0p/(1-ek0p) + ekym*ek0m/(1-ek0m)  ) )
                Gyzs = 1j/(2*a*k ** 2)*phase*( - ky*np.exp(1j*kz*Z) 
                    + np.sum( - ekzmky*kym - ekzmmky*kymm 
                    + 2*ky*np.exp(-km*Z)*np.cos(km*Y + 1j*ky*Z) + 2j*km*np.exp(-km*Z)*np.sin(km*Y + 1j*ky*Z))
                    - 1*ky*(ekyp*ek0p/(1-ek0p) + ekym*ek0m/(1-ek0m) ) 
                    + 1*2*np.pi/a*(  ekyp*ek0p/(1 - ek0p) ** 2 - ekym*ek0m/(1 - ek0m) ** 2 )) 

                Gzxs = kx/k*tzx
                GxyEMs = tzx

    gf = np.zeros((6, 6) , dtype = 'complex_')

    gf[0,0] =  gf[3,3] = Gxxs
    gf[1,1] =  gf[4,4] = Gyys
    gf[2,2] =  gf[5,5] = Gzzs
    gf[0,1] =  gf[1,0] = gf[3,4] =  gf[4,3] = Gxys
    gf[1,2] =  gf[2,1] = gf[4,5] =  gf[5,4] = Gyzs*np.sign(z0-z)
    gf[2,0] =  gf[0,2] = gf[5,3] =  gf[3,5] = Gzxs*np.sign(z0-z)

    gf[0,4] =  gf[4,0] = -GxyEMs*np.sign(z0-z)
    gf[1,3] =  gf[3,1] = GxyEMs*np.sign(z0-z)
    gf[1,5] =  gf[5,1] = -GyzEMs
    gf[2,4] =  gf[4,2] = GyzEMs
    gf[2,3] =  gf[3,2] = -GzxEMs
    gf[0,5] =  gf[5,0] = GzxEMs

    return gf
        
