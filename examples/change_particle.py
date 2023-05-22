# Import libraries
import cemd_metasurf as cemd
import numpy as np
import matplotlib.pyplot as plt

# Set a rectangular metasurface of lattice constant a and b.
a = 400
b = 300
my_metasurface = cemd.Metasurface(a,b)

# Change the property of the particle that defined the unit cell
my_particle = cemd.ParticleMie(r_p = 80, eps = 9, wp=0, wr=0, gr=0)
my_metasurface.set_particles(my_particle)

# Calculation of R and T for a given incidence
nk = 1001
theta = 10*np.pi/180
k = np.linspace(0.45,0.85,nk)*2*np.pi/a
ky = np.zeros_like(k)
kx = k*np.sin(theta) 
my_bloch = cemd.BlochWavevector(k,kx,ky)

my_metasurface.calc_gb_kxky(my_bloch)
my_metasurface.calc_rt_kxky()
r_tm, r_te, t_tm, t_te = my_metasurface.get_rt()

plt.plot(k*a/2/np.pi,r_te, label='TE')
plt.plot(k*a/2/np.pi,r_tm, label='TM')
plt.title('Reflectance at $theta$ = ' + str(theta*180/np.pi))
plt.xlabel('ka/2pi')
plt.ylabel('R')
plt.legend()
plt.show()
