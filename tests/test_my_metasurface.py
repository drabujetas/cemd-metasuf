import unittest
import numpy as np
from cemd_metasurf.my_metasurface import Metasurface

class MetasurfaceTestClass(unittest.TestCase):
	
	def setUp(self):
		a = 400
		b = a
		self.metasurface = Metasurface(a,b)

	def test_lattice(self):
		a_m, b_m, th_m = self.metasurface.get_lattice()
		self.assertEqual(a_m,400)
		self.assertEqual(b_m,400)
		self.assertEqual(th_m,np.pi/2)

	def test_lattice(self):
		x, y, z = self.metasurface.get_unit_cell()
		self.assertEqual(x,0)
		self.assertEqual(y,0)
		self.assertEqual(z,0)

	def test_particle(self):
		particle = self.metasurface.particles[0]
		self.assertEqual(particle.r_p,400/4)
