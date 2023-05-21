# Import libraries
import cemd_metasurf as cemd
import numpy as np
import matplotlib.pyplot as plt

# Set a square metasurface of lattice constant a.
a = 400
my_metasurface = cemd.Metasurface(a,a)

# Set the wavevector at which the reflectance (R) is calculated
# theta is the angle of incidence
nk = 1001
theta = 10*np.pi/180
k = np.linspace(0.45,0.85,nk)*2*np.pi/a
ky = np.zeros_like(k)
kx = k*np.sin(theta) 
my_bloch = cemd.BlochWavevector(k,kx,ky)

# Calculation of the depolarization Green function and R
my_metasurface.calc_gb_kxky(my_bloch)
my_metasurface.calc_rt_kxky()
r_tm, r_te, t_tm, t_te = my_metasurface.get_rt()

# Ploting R
plt.plot(k*a/2/np.pi,r_te, label='TE')
plt.plot(k*a/2/np.pi,r_tm, label='TM')
plt.title('Reflectance at $theta$ = ' + str(theta*180/np.pi))
plt.xlabel('ka/2pi')
plt.ylabel('R')
plt.legend()
plt.show()
