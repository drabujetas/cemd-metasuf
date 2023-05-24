"""
Here the class ReTr for calculating refletion and tranmission can be found.
"""

import numpy as np	
from . import rt_functions as rtf

class ReTr(object):

	def calc_rt_kxky(self, my_bloch = None, ind_ini = 0, n=0,m=0):
		"""
		Calculates the reflectance (R) and transmitance (T) for a transverse electric (TE) or 
		transverse magnetic (TM) incoming plane wave at all Bloch waves stored in array_k_gb.
		
		If a my_bloch is given, R and T will be calculated over this values, by calculating first
		the depolarization Green function (gb). Consider that in this case, the value of ind_ini
		is set to the size of the previous self.array_k_gb.shape, in order of only calculating
		R and T for the given my_bloch.
		Also, clear.array_k_rt() and clear.array_k_gb() are called at the end if my_bloch is given.
		
		The values are stored in "array_k_rt", with dimension along "axis = 1" (columns) are:
			1. k
			2. kx
			3. ky
			4. n
			5. m
			6. r_tm
			7. r_te
			8. t_tm
			9. t_te

		:param my_bloch: The object with the information of the wavevectos. By default it is not needed, using the wavevetors stored in self.array_k_gb. If a my_bloch is used, R and T would be calculated over these values.
		:type my_bloch: classes.BlochWavevector
		:param ind_ini: From which value of the stored gb start to calculate R and T (try to generilized to a range). Only used is "my_bloch = None".
		:type ind_ini: Int 
		:param n: Diffractive order along x-axis.
		:type n: Int
		:param m: Diffractive order along the other axis defined by the lattice.
		:type m: Int
		"""
		x, y, z = self.get_unit_cell()
		if type(my_bloch) != type(None):
			if type(self.array_k_gb) != type(None):
				ind_ini = self.array_k_gb.shape[0]
			self.calc_gb_kxky(my_bloch, append_k_gb = True)
		for i in range(ind_ini,self.array_k_gb.shape[0]):
			self.k, self.kx, self.ky = self.array_k_gb[i,0:3].real
			self.set_alpha()
			if type(x) == np.float64:
				self.gb_kxky = self.array_k_gb[i,3:].reshape(6,6)
				self.rt_kxky_nm = rtf.calc_rt(self,n,m)
			else:
				return print("the case of complex unit cells are not implemented")
			k_rt = np.append([self.k,self.kx,self.ky,n,m],self.rt_kxky_nm.reshape(1,-1)).reshape(1,-1)
			if type(self.array_k_rt) == type(None):
				self.array_k_rt = k_rt
			else:
				self.array_k_rt = np.append(self.array_k_rt,k_rt, axis=0)
		if type(my_bloch) != type(None):
			self.clean_array_k_gb()
			self.clean_array_k_rt()
	

	def clear_array_k_rt(self):
		"""
		Clear the stored reflectance and transmitance.
		"""
		self.array_k_rt = None

	def clean_array_k_rt(self):
		"""
		Clean the stored reflectance and transmitance.
		Remove repeted rows and sort by k -> kx -> ky -> n -> m.
		"""
		sort_array_k_rt = self.array_k_rt[ np.lexsort( (self.array_k_rt[:,0], self.array_k_rt[:,1], self.array_k_rt[:,2], self.array_k_rt[:,3], self.array_k_rt[:,4] ) ) ]
		self.array_k_rt = np.unique(sort_array_k_rt, axis=0) 

	def get_rt(self):
		"""
		Return reflectance and transmitance.
		For konwing "(k, kx, ky)" and "(n, m)" look self.array_k_rt.

		:return: tuple with R and T 
		"""
		r_tm = self.array_k_rt[:,5]
		r_te = self.array_k_rt[:,6]
		t_tm = self.array_k_rt[:,7]
		t_te = self.array_k_rt[:,8]
		return r_tm, r_te, t_tm, t_te

	def calc_rt_complex_kxky(self, my_bloch = None, ind_ini = 0):
		"""
		Calculates the complex specular reflection (r) and transmissio (t) for a transverse electric (TE) or 
		transverse magnetic (TM) incoming plane wave at all Bloch waves stored in array_k_gb.
		
		If a my_bloch is given, r and t will be calculated over this values, by calculating first
		the depolarization Green function (gb). Consider that in this case, the value of ind_ini
		is set to the size of the previous self.array_k_gb.shape, in order of only calculating
		r and t for the given my_bloch.
		Also, clear.array_k_rt_complex() and clear.array_k_gb() are called at the end if my_bloch is given.
		
		The values are stored in "array_k_rt_complex", with dimension along "axis = 1" (columns) are:
			1. k
			2. kx
			3. ky
			4. r_tm
			5. r_te
			6. t_tm
			7. t_te

		:param my_bloch: The object with the information of the wavevectos. By default it is not needed, using the wavevetors stored in self.array_k_gb. If a my_bloch is used, R and T would be calculated over these values.
		:type my_bloch: classes.BlochWavevector
		:param ind_ini: From which value of the stored gb start to calculate R and T (try to generilized to a range). Only used is "my_bloch = None".
		:type ind_ini: Int 
		:param n: Diffractive order along x-axis.
		"""
		x, y, z = self.get_unit_cell()
		if type(my_bloch) != type(None):
			if type(self.array_k_gb) != type(None):
				ind_ini = self.array_k_gb.shape[0]
			self.calc_gb_kxky(my_bloch, append_k_gb = True)
		for i in range(ind_ini,self.array_k_gb.shape[0]):
			self.k, self.kx, self.ky = self.array_k_gb[i,0:3].real
			self.set_alpha()
			if type(x) == np.float64:
				self.gb_kxky = self.array_k_gb[i,3:].reshape(6,6)
				self.rt_kxky_complex = rtf.calc_rt_complex(self)
			else:
				return print("the case of complex unit cells are not implemented")
			k_rt = np.append([self.k,self.kx,self.ky],self.rt_kxky_complex.reshape(1,-1)).reshape(1,-1)
			if type(self.array_k_rt_complex) == type(None):
				self.array_k_rt_complex = k_rt
			else:
				self.array_k_rt_compex = np.append(self.array_k_rt_complex,k_rt, axis=0)
		if type(my_bloch) != type(None):
			self.clean_array_k_gb()
			self.clean_array_k_rt_complex()

	def clear_array_k_rt_complex(self):
		"""
		Clear the stored reflectance and transmitance.
		"""
		self.array_k_rt_complex = None

	def clean_array_k_rt_complex(self):
		"""
		Clean the stored reflectance and transmitance.
		Remove repeted rows and sort by k -> kx -> ky.
		"""
		sort_array_k_rt_complex = self.array_k_rt_complex[ np.lexsort( (self.array_k_rt_complex[:,0], self.array_k_rt_complex[:,1], self.array_k_rt_complex[:,2]) ) ]
		self.array_k_rt_complex = np.unique(sort_array_k_rt_complex, axis=0) 

	def get_rt_complex(self):
		"""
		Return reflectance and transmitance.
		For konwing "(k, kx, ky)" look self.array_k_rt_complex.

		:return: tuple with r and t 
		"""
		r_tm = self.array_k_rt_complex[:,3]
		r_te = self.array_k_rt_complex[:,4]
		t_tm = self.array_k_rt_complex[:,5]
		t_te = self.array_k_rt_complex[:,6]
		return r_tm, r_te, t_tm, t_te


	def calc_rt_pol_kxky(self, my_bloch = None, ind_ini = 0, pol_tm = 1, pol_te = 1j, n=0,m=0):
		"""
		Calculates the de reflectance (R) and transmitance (T) at all Bloch waves stored in array_k_gb
		for a incoming plane wave with polarization defiend by the parameters pol_tm and pol_te:

		:math:`\phi_{0} = \phi_{TM}\mathrm{pol_tm} + \phi_{TE}\mathr{pol_te}`.
		
		Thus, by default the incident is a circular polarized wave (pol_tm = 1, pol_te = 1j).

		If a my_bloch is given, R and T will be calculated over this values, by calculating first
		the depolarization Green function (gb). Consider that in this case, the value of ind_ini
		is set to the size of the previous self.array_k_gb.shape, in order of only calculating
		R and T for the given my_bloch.
		Also, clear.array_k_rt_pol() and clear.array_k_gb() are called at the end if my_bloch is given.
		
		The values are stored in "array_k_rt", with dimension along "axis = 1" (columns) are:
			1. k
			2. kx
			3. ky
			4. pol_tm
			5. pot_te
			6. n
			7. m
			8. r_pol
			9. t_pol

		:param my_bloch: The object with the information of the wavevectos. By default it is not needed, using the wavevetors stored in self.array_k_gb. If a my_bloch is used, R and T would be calculated over these values.
		:type my_bloch: classes.BlochWavevector
		:param ind_ini: From which value of the stored gb start to calculate R and T (try to generilized to a range). Only used is "my_bloch = None".
		:type ind_ini: Int 
		:param pol_tm: TM amplitude.
		:type pol_tm: complex
		:param pol_te: TE amplitude.
		:type pol_te: complex
		:param n: Diffractive order along x-axis.
		:type n: Int
		:param m: Diffractive order along the other axis defined by lattice.
		:type m: Int
		"""
		x, y, z = self.get_unit_cell()
		if type(my_bloch) != type(None):
			if type(self.array_k_gb) != type(None):
				ind_ini = self.array_k_gb.shape[0]
			self.calc_gb_kxky(my_bloch, append_k_gb = True)
		for i in range(ind_ini,self.array_k_gb.shape[0]):
			self.k, self.kx, self.ky = self.array_k_gb[i,0:3].real
			self.set_alpha()
			if type(x) == np.float64:
				self.gb_kxky = self.array_k_gb[i,3:].reshape(6,6)
				self.rt_kxky_nm = rtf.calc_rt_pol(self,pol_tm,pol_te,n,m)
			else:
				return print("the case of complex unit cells are not implemented")
			k_rt = np.append([self.k,self.kx,self.ky,pol_tm,pol_te,n,m],self.rt_kxky_nm.reshape(1,-1)).reshape(1,-1)
			if type(self.array_k_rt_pol) == type(None):
				self.array_k_rt_pol = k_rt
			else:
				self.array_k_rt_pol = np.append(self.array_k_rt_pol,k_rt, axis=0)
		if type(my_bloch) != type(None):
			self.clean_array_k_gb()
			self.clean_array_k_rt_pol()
	

	def clear_array_k_rt_pol(self):
		"""
		Clear the stored reflectance and transmitance.
		"""
		self.array_k_rt_pol = None

	def clean_array_k_rt_pol(self):
		"""
		Clean the stored reflectance and transmitance.
		Remove repeted rows and sort by k -> kx -> ky -> pol_tm -> pol_te -> n -> m.
		"""
		sort_array_k_rt_pol = self.array_k_rt_pol[ np.lexsort( (self.array_k_rt_pol[:,0], self.array_k_rt_pol[:,1], self.array_k_rt_pol[:,2], self.array_k_rt_pol[:,3], self.array_k_rt_pol[:,4], self.array_k_rt_pol[:,5], self.array_k_rt_pol[:,6] ) ) ]
		self.array_k_rt_pol = np.unique(sort_array_k_rt_pol, axis=0) 

	def get_rt_pol(self):
		"""
		Return reflectance and transmitance.
		For konwing "(k, kx, ky)", "(pol_tm, pol_te)" and "(n, m)" look self.array_k_rt_pol.

		:return: tuple with R and T 
		"""
		r_pol = self.array_k_rt_pol[:,7]
		t_pol = self.array_k_rt_pol[:,8]
		return r_pol, t_pol