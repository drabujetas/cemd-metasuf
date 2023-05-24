# cemd_metasurf
Coupled electric and magnetic dipole (CEMD) library for calculating optical properties of metasurfaces.

### Installation

Install cemd-metasurf [from PyPI](https://pypi.org/project/cemd-metasurf/) with

```
pip install cemd-metasurf
```

The current version can calculate reflection and transmissionn for lattices with one particle per unit cell. In future releases it is planning to add new functionalities as:

- More than one particle per unit cell.
- Near field calculation (with periodicity along the lattice).
- Local density of states and near field for dipole excitation.
- Presence of layered substrate.

For any comment feel free to contact me!
drabujetas@gmail.com

The documentation can be found at
https://cemd-metasuf.readthedocs.io/

### Examples

The next two examples calculate the reflectance of a square array with the same parameters as used in [D.R. Abujetas, et. al., PRB, 102, 125411 (2020)].

- 1. Reflectance at a constant angle of incidence.
```Python
# Load libraries
import cemd_metasurf as cemd
import numpy as np
import matplotlib.pyplot as plt

# Set a square metasurface of lattice constant a.
a = 400
my_metasurface = cemd.Metasurface(a,a)

# Set the wavevector at which the reflectance (R) is calculated.
# Theta is the angle of incidence.
nk = 1001
theta = 10*np.pi/180
k = np.linspace(0.45,0.85,nk)*2*np.pi/a
ky = np.zeros_like(k)
kx = k*np.sin(theta) 
my_bloch = cemd.BlochWavevector(k,kx,ky)

# Calculation of the depolarization Green function and R.
my_metasurface.calc_gb_kxky(my_bloch)
my_metasurface.calc_rt_kxky()
r_tm, r_te, t_tm, t_te = my_metasurface.get_rt()

# Plotting R.
plt.plot(k*a/2/np.pi,r_te, label='TE')
plt.plot(k*a/2/np.pi,r_tm, label='TM')
plt.title('Reflectance at $theta$ = ' + str(theta*180/np.pi))
plt.xlabel('ka/2pi')
plt.ylabel('R')
plt.legend()
plt.show()
```
It is also possible to directly call ".calc_rt_kxky()" with the list of wavevectors:
```Python
my_metasurface.calc_rt_kxky(my_bloch)
```
and the depolarization Green function will be first calculated over these wavevectors.

- 2. 2D map of the reflectance for TE polarization as a function of the frequency and the angle of incidence.
```Python
# Load libraries
import cemd_metasurf as cemd
import numpy as np
import matplotlib.pyplot as plt

# Set a square metasurface of lattice constant a.
a = 400
my_metasurface = cemd.Metasurface(a,a)

# Set the wavevector at which the reflectance (R) is calculated.
nk = 101
ang1 = np.linspace(0,89,nk)*np.pi/180
k = np.linspace(0.3,0.9,nk)*2*np.pi/a
ky = np.zeros_like(k)

# Calculation of the depolarization Green function and R.
for ang in ang1:
    kx = k*np.sin(ang) 
    my_bloch = cemd.BlochWavevector(k,kx,ky)
    my_metasurface.calc_gb_kxky(my_bloch)
my_metasurface.calc_rt_kxky()
r_tm, r_te, t_tm, t_te = my_metasurface.get_rt()
r_te = r_te.reshape(nk,nk)

# Plotting R.
fig, ax = plt.subplots()
CS = ax.contourf(k*a/2/np.pi,ang1*180/np.pi,r_te,100)
fig.suptitle('2D map of the reflectance for TE waves')
ax.set_xlabel('ka/2pi')
ax.set_ylabel('$theta$ (deg)')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel("R")
plt.show()
```

The properties of the particle are automatically set to a dielectric sphere of permittivity "eps = 12.25" a radius "r_p = a/4" (un quarter of the lattice constant), where the polarizability is calculated using the Mie coefficients. This can be check it by
```Python
print(my_metasurface.particles[0].ei)
print(my_metasurface.particles[0].r_p)
```
where "self.particles" is a list where each entry is one particle of the unit cell (by default there is only one particle). The properties of the particle can be modified by defining a new particle as
```Python
# Definition of a particle of radius "r_p = 80" and permittivity "eps = 9"
my_particle = cemd.ParticleMie(r_p = 80, eps = 9)
# Set the properties of the particle in the metasurface. "set_particle" set all particles with the same "my_particle"
my_metasurface.set_particles(my_particle)
# The next statement modifies only the properties of particle "i". In this case, both lines do the same 
my_metasurface.set_particle_i(my_particle,i=0)
# Check its properties 
print(my_metasurface.particles[0].ei)
print(my_metasurface.particles[0].r_p)
```
In addition, the permittivity can be described by Drude-Lorentz model
```Python
my_particle = cemd.ParticleMie(r_p = 80, eps = 9, wp=0, wr=0, gr=0)
```
where "wp", "wr" and "gr" are the plasma frequency, resonant frequency and width of the resonance (in units of frequency, "w = kc" where "c" is the speed of light).  
There is another class "ParticleUserDefined" in which tabulated values of the polarizability can be directly given, and more particles can be included in future releases.
