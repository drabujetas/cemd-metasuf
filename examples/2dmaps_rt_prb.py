# Import libraries
import cemd_metasurf as cemd
import numpy as np
import matplotlib.pyplot as plt

# Set a square metasurface of lattice constant a
a = 400
my_metasurface = cemd.Metasurface(a,a)

# Set the wavevector at which the reflectance (R) is calculated
nk = 101
ang1 = np.linspace(0,89,nk)*np.pi/180
k = np.linspace(0.3,0.9,nk)*2*np.pi/a
ky = np.zeros_like(k)

# Calculation of the depolarization Green function and R
for ang in ang1:
    kx = k*np.sin(ang) 
    my_bloch = cemd.BlochWavevector(k,kx,ky)
    my_metasurface.calc_gb_kxky(my_bloch)
my_metasurface.calc_rt_kxky()
r_tm, r_te, t_tm, t_te = my_metasurface.get_rt()
r_te = r_te.reshape(nk,nk)

# Ploting R
fig, ax = plt.subplots()
CS = ax.contourf(k*a/2/np.pi,ang1*180/np.pi,r_te,100)
fig.suptitle('2D map of the reflectance for TE waves')
ax.set_xlabel('ka/2pi')
ax.set_ylabel('$theta$ (deg)')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("R")
plt.show()
