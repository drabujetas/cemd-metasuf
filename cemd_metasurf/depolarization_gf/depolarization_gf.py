
import numpy as np	
from . import depolarization_functions as dgf

class DGreenFunction(object):

	def calc_gb_kxky(self,my_bloch = None, append_k_gb = True):
		"""
		Calculates the depolarization Green functions (gb) for the lattice at the given Bloch wavevectors.

		If my_bloch is not provided, gb would be calculated by using the wavevector configuration of 
		the lattice and the clean function will be called.

		:param my_bloch: The object with the information of the wavevectos at which gb is calculated.
		:type my_bloch: classes.BlochWavevector 
		:param append_k_gb: Set if the values are stored in the self.array_k_gb.
		:type append_k_gb: Bool
		"""
		x, y, z = self.get_unit_cell()
		if type(my_bloch) != type(None):
			kb, kxb, kyb = my_bloch.get_bloch()
			for i in range(kb.shape[0]):
				self.k, self.kx, self.ky = kb[i], kxb[i], kyb[i]
				if type(x) == np.float64:
					self.gb_kxky = dgf.calc_gb_1puc(self)
				else:
					self.gb_kxky = dgf.calc_gb_npuc(self)
				if append_k_gb:
					k_gb = np.append([self.k,self.kx,self.ky],self.gb_kxky.reshape(1,-1)).reshape(1,-1)
					if type(self.array_k_gb) == type(None):
						self.array_k_gb = k_gb
					else:
						self.array_k_gb = np.append(self.array_k_gb,k_gb, axis=0)
		else:
			if type(x) == np.float64:
				self.gb_kxky = dgf.calc_gb_1puc(self)
			else:
				self.gb_kxky = dgf.calc_gb_npuc(self)	
			if append_k_gb:
				k_gb = np.append([self.k,self.kx,self.ky],self.gb_kxky.reshape(1,-1)).reshape(1,-1)
				if type(self.array_k_gb) == type(None):
					self.array_k_gb = k_gb
				else:
					self.array_k_gb = np.append(self.array_k_gb,k_gb, axis=0)
			self.clean_array_k_gb()


	def clear_array_k_gb(self):
		"""
		Clear the stored depolarization Green functions.
		"""
		self.array_k_gb = None

	def clean_array_k_gb(self):
		"""
		Clean the stored depolarization Green functions.
		Remove repeted rows and sort by k -> kx -> ky.
		"""
		sort_array_k_gb = self.array_k_gb[ np.lexsort( (self.array_k_gb[:,0], self.array_k_gb[:,1], self.array_k_gb[:,2] ) ) ]
		self.array_k_gb = np.unique(sort_array_k_gb, axis=0) 