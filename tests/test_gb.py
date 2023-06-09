import unittest
import numpy as np
from cemd_metasurf.my_metasurface import Metasurface
from cemd_metasurf.my_metasurface import BlochWavevector
from cemd_metasurf.depolarization_gf.depolarization_gf import DGreenFunction

class DGreenFunctionTestClass(unittest.TestCase):
	
	def setUp(self):
		a = 400
		b = a
		self.metasurface = Metasurface(a,b)
		k = np.linspace(0.1,0.99,100)*2*np.pi/a
		ky = np.zeros_like(k)
		kx = k*np.sin(50*np.pi/180) 
		my_bloch = BlochWavevector(k,kx,ky)
		self.metasurface.calc_gb_kxky(my_bloch)

	def test_clear_gb(self):
		self.metasurface.clear_array_k_gb()
		self.assertEqual(type(self.metasurface.array_k_gb), type(None))

	def test_clean_gb(self):
		self.metasurface.clean_array_k_gb()
