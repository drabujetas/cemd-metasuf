import numpy as np
from .polarizability import mie_functions
from .depolarization_gf.depolarization_gf import DGreenFunction
from .reflec_transm.reflec_transm import ReTr

class Metasurface(DGreenFunction, ReTr):
	"""
	Class for defining the metasurface
	"a" and "b" are the lengths of the lattice vectors ("a" along the x-axis) and "th" is the angle between the lattice vectors.
	For example the lattice vector "hat(v_1)" and "hat(v_2)" would be:
	"hat(v_1) = a hat(x)", 
	"hat(v_2) = b (cos(th)*hat(x) + sin(th)*hat(y))".
	Note that "|hat(v_1)| = a" and "|hat(v_2)| = b".
	"x", "y" and "z" defined the position of the particles in the unit cell.
	"eps_b" is the backgroud permittivity and the permittivity of the layred substrate (if present).
	"d_layer" defined the with of the layered substrate (if present).

	:param a: Length lattice vector along the x-axis
	:type a: float
	:param b: Length lattice vector along the other axis (defined by the :param th)
	:type b: float
	:param x: Position in the x-axis of the particles that composed the unit cell.
	:type x: float or numpy.ndarray
	:param y: Position in the y-axis of the particles that composed the unit cell.
	:type y: float or numpy.ndarray
	:param z: Position in the z-axis of the particles that composed the unit cell.
	:type z: float or numpy.ndarray
	:param th: Angle between lattice vectors.
	:type th: float
	:param eps_b: Permittivity of the media.
	:type esp_b: float (and numpy.ndarray in the future)
	:param d_layer: It will defined the width of the layers of the substrate.
	:type d_later: float (and numpy.ndarray in the future)
	"""
	def __init__(self,a=400,b=400,x=0,y=0,z=0,th=np.pi/2,eps_b = 1, d_layer = 0):
		if isinstance(x,int) or isinstance(x,float):
			x = np.float64(x)
			y = np.float64(y)
			z = np.float64(z)
			self.__num_part = 1
		elif len(x) == 1:
			x = np.float64(x[0])
			y = np.float64(y[0])
			z = np.float64(z[0])
			self.__num_part = 1
		else:
			self.__num_part = np.max(x.shape)
			if y.shape != x.shape or y.shape != z.shape or z.shape != y.shape:
				raise ValueError("Size mismatch between x, y and z")
		self.__a = a
		self.__b = b
		self.__x = x
		self.__y = y
		self.__z = z
		self.__th = th
		self.__eps_b = eps_b
		self.__d_layer = d_layer
		self.particles = [ParticleMie(a/4,eps = 3.5**2)]*self.__num_part
		self.gb_kxky = None
		self.k = None
		self.kx = None
		self.ky = None
		self.array_k_gb = None
		self.array_k_rt = None

	def set_lattice(self,a,b,th):
		"""
		Set the properties of the lattice (lattice vectors and the angle between them).

		:param a: Length lattice vector along the x-axis
		:type a: float
		:param b: Length lattice vector along the other axis (defined by the :param th)
		:type b: float
		:param th: Angle between lattice vectors.
		:type th: float
		"""
		self.__a = a
		self.__b = b
		self.__th = th

	def get_lattice(self):
		"""
		Return lattice parameters.

		:return: tuple with the lattice parameters a, b and th.
		"""
		return self.__a, self.__b, self.__th

	def set_unit_cell(self,x,y,z):
		"""
		Set the position of the particles in the unit cell.

		:param x: Position in the x-axis of the particles that composed the unit cell.
		:type x: float or numpy.ndarray
		:param y: Position in the y-axis of the particles that composed the unit cell.
		:type y: float or numpy.ndarray
		:param z: Position in the z-axis of the particles that composed the unit cell.
		:type z: float or numpy.ndarray
		"""
		if isinstance(x,int) or isinstance(x,float):
			x = np.float64(x)
			y = np.float64(y)
			z = np.float64(z)
			self.__num_part = 1
		elif len(x) == 1:
			x = np.float64(x[0])
			y = np.float64(y[0])
			z = np.float64(z[0])
			self.__num_part = 1
		else:
			self.__num_part = np.max(x.shape)
			if y.shape != x.shape or y.shape != z.shape or z.shape != y.shape:
				raise ValueError("Size mismatch between x, y and z")

		self.__x = x
		self.__y = y
		self.__z = z

	def get_unit_cell(self):
		"""
		Return unit cell configuration.

		:return: tuple with the position of the particles in the unit cell (x, y and z)
		"""
		return self.__x, self.__y, self.__z

	def set_bloch(self,my_bloch):
		"""
		Set the actual Bloch wavevector configuration (it takes the last entry of my_bloch).
		"k" is the wavevector in the medium where the metasurface is placed (k = 2pi/lambda*sqrt(eps_b[0]))
		By the moment, "eps_b" is a number, but it will be generized in the future for supporting substrates.
		"kx" and "ky" are the Bloch wavevector (periodicity along the metasurface plane).

		:param my_bloch: The object with the information of the wavevectos
		:type my_bloch: classes.BlochWavevector
		"""
		k, kx, ky = my_bloch.get_bloch()
		self.k, self.kx, self.ky = k[-1], kx[-1], ky[-1]

	def get_bloch(self):
		"""
		Return the actual Bloch wave configuration.

		:return: tuple with the Bloch wave configuration (k, kx and ky)
		"""
		return self.k, self.kx, self.ky

	def set_particles(self,my_particle):
		"""
		Set all particle to have the properties given by "my_particle".

		:param my_particle: The object with the information of the particle.
		:type my_particle: classes.ParticleMie (or other particle)
		"""
		self.particles = [my_particle]*self.__num_part

	def set_particle_i(self,my_particle,i):
		"""
		Set the particle "i" to have the properties given by "my_particle".

		:param my_particle: The object with the information of the particle.
		:type my_particle: classes.ParticleMie (or other particle).
		:param i: Index of the particle that will be set.
		:type i: Int
		"""
		if i > len(self.particles) or i < 0:
			raise ValueError("i bigger than the number of particles (or negative)")
		else:
			self.particles[i] = my_particle

	def set_alpha(self):
		"""
		Set the polarizability to the one defined in self.particles at the inner wavevector (self.k).
		"""
		alp = np.zeros((self.__num_part*6,self.__num_part*6), dtype = "complex_")
		i = 0
		for particle in self.particles:
			particle.set_alpha(self.k,self.__eps_b)
			alp[i*6:i*6+6] = particle.alp
			i = i+1
		self.alp_uc = alp

	def set_alpha_user_defined(self,alp):
		"""
		Set the polarizability to a given specific matrix.
		At the moment is not really used ("set_alpha" is the one used for setting the polarizability 
		for calculating r and t), but in the future I would like to give more flexibility to the code
		and, for example, being able of easyly set quirality (modify the off-diagonal matrices of alp).

		:param alp: Polarizability of the system.
		:type alp: numpy.ndarray
		"""
		if alp.shape != (self.__num_part*6, self.__num_part*6):
			raise ValueError("the dimension of alp is not the appropieted") 
		self.alp_uc = alp


class BlochWavevector:
	"""
	Class to handle the Bloch wavevectors
	"k" is the wavevector (in the medium).
	"kx" and "ky" are the Bloch wavevector (periodicity along unit cells).

	:param k: Wavevector in the medium.
	:type k: float or numpy.ndarray
	:param kx: Bloch wavevector (Floquet periodicity) along the x-axis.
	:type kx: float or numpy.ndarray
	:param ky: Bloch wavevector (Floquet periodicity) along the y-axis.
	:type ky: float or numpy.ndarray
	"""
	def __init__(self,k,kx,ky):
		if (isinstance(k,int) or isinstance(k,float)):
			self.__k = np.array([k])
			self.__kx = np.array([kx])
			self.__ky = np.array([ky])
		else:
			self.__k = k.reshape(-1,)
			self.__kx = kx.reshape(-1,)
			self.__ky = ky.reshape(-1,)

	def get_bloch(self):
		"""
		Get the wavevectors.

		:return: tuple with the Bloch wavevectors (k, kx and ky)
		"""
		return self.__k, self.__kx, self.__ky

	def get_bloch_i(self,i):
		"""
		Get the wavevector at index "i"

		:param i: Index of the particle that will be set.
		:type i: Int 
		:return: tuple with the Bloch wavevectors (k, kx and ky)
		"""
		if i < self.__k.shape[0]:
			return self.__k[i], self.__kx[i], self.__ky[i]
		else:
			return print("i bigger than the dimension of the wavevectors")

class ParticleMie:
	"""
	Class to set the properties of the particle to a Mie particle with a permittivity
	given by a Drude-Lorentz model (at the moment with one single resonance).

	:param r_p: Radius of the particles.
	:type r_p: Float
	:param ei: Permittivity at high frecuencies (avobe resonances).
	:type ei: Float
	:param wp: Plasma frequency.
	:type wp: Float (I will generalize it to an array setting several resonances)
	:param wr: Resonant frequency.
	:type wr: Float (I will generalize it to an array setting several resonances)
	:param gr: Width of the resonance.
	:type gr: Float (I will generalize it to an array setting several resonances)
	"""
	def __init__(self,r_p=1,eps=1,wp=0, wr=0, gr=0):
		self.ei = eps
		self.wp = wp
		self.wr = wr
		self.gr = gr
		self.r_p = r_p

	"""
	Set the polatizability of the particle.

	:param k: Wavevector (in the media).
	:type k: Float 
	:param eps_b: Permittivity of the media.
	:type eps_b: Float 
	"""
	def set_alpha(self,k, eps_b=1):
		k0 = k/np.sqrt(eps_b)
		w = k0*3e8
		eps = self.ei + self.wp**2/(self.wr**2 - w**2 - 1j*w*self.gr)
		alp = mie_functions.get_alpha_mie(k0,self.r_p,np.sqrt(eps),np.sqrt(eps_b))
		self.alp = alp


class ParticleUserDefined:
	"""
	Class to set the properties of the particle to tabulated data. 
	
	It will develop in future releases.
	"""
	def __init__(self,k_array,alp_e,alp_m,alp_chiral = None):
		if k_array.size != k_array.shape[0]:
			ValueError("k_array must be a one dimensional array")
		self.k_array = k_array
		self.alp_e = alp_e
		self.alp_m = alp_m

	"""
	Set the polatizability of the particle at k.
	"""
	def set_alpha(self,k, eps_b=1):
		size_k = self.k_array.size
		alp_e = self.alp_e.reshape(size_k,-1)
		alp_m = self.alp_m.reshape(size_k,-1)
		if alp_e.shape[1] != alp_m.shape[1]:
			ValueError("alp_e and alp_m must have the same dimensionality")
		alp = np.zeros((6,6), dtype = 'complex_')
		if alp_e.shape[1] == 1:
			alpha_e = np.interp(k,self.k_array,alp_e)
			alpha_m = np.interp(k,self.k_array,alp_m)
			id3 = np.eye(3)  
			alp[0:3,0:3] = id3*alpha_e
			alp[3:6,3:6] = id3*alpha_m
		elif alp_e.shape[1] == 3:
			alp[0,0] = np.interp(k,self.k_array,alp_e[:,0])
			alp[1,1] = np.interp(k,self.k_array,alp_e[:,1])
			alp[2,2] = np.interp(k,self.k_array,alp_e[:,2])
			alp[3,3] = np.interp(k,self.k_array,alp_m[:,0])
			alp[4,4] = np.interp(k,self.k_array,alp_m[:,1])
			alp[5,5] = np.interp(k,self.k_array,alp_m[:,2])
		else:
			print("not implemented")

		if type(self.alp_chiral) != type(None):
			print("not implemented")



