import unittest
import numpy as np
from cemd_metasurf.my_metasurface import Metasurface
from cemd_metasurf.my_metasurface import BlochWavevector
from cemd_metasurf.reflec_transm.reflec_transm import ReTr

class ReTrTestClass(unittest.TestCase):
	
	def setUp(self):
		a = 400
		b = a
		self.metasurface = Metasurface(a,b)
		k = np.linspace(0.1,0.99,100)*2*np.pi/a
		ky = np.zeros_like(k)
		kx = k*np.sin(50*np.pi/180) 
		my_bloch = BlochWavevector(k,kx,ky)
		self.metasurface.calc_rt_kxky(my_bloch)
		self.metasurface.calc_rt_complex_kxky(my_bloch)
		self.metasurface.calc_rt_pol_kxky(my_bloch)

	def test_get_rt(self):
		r_tm, r_te, t_tm, t_te = self.metasurface.get_rt()

	def test_clear_rt(self):
		self.metasurface.clear_array_k_rt()
		self.assertEqual(type(self.metasurface.array_k_rt), type(None))

	def test_clean_rt(self):
		self.metasurface.clean_array_k_rt()

	def test_get_rt_complex(self):
		r_tm, r_te, t_tm, t_te = self.metasurface.get_rt_complex()

	def test_clear_rt_complex(self):
		self.metasurface.clear_array_k_rt_complex()
		self.assertEqual(type(self.metasurface.array_k_rt_complex), type(None))

	def test_clean_rt_complex(self):
		self.metasurface.clean_array_k_rt_complex()

	def test_get_rt_pol(self):
		r_tm, t_pol = self.metasurface.get_rt_pol()

	def test_clear_rt_pol(self):
		self.metasurface.clear_array_k_rt_pol()
		self.assertEqual(type(self.metasurface.array_k_rt_pol), type(None))

	def test_clean_rt_pol(self):
		self.metasurface.clean_array_k_rt_pol()