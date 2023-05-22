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
		self.metasurface.set_lattice(300,500,1.3)
		a_m, b_m, th_m = self.metasurface.get_lattice()
		self.assertEqual(a_m,300)
		self.assertEqual(b_m,500)
		self.assertEqual(th_m,1.3)

	def test_unit_cell(self):
		x, y, z = self.metasurface.get_unit_cell()
		self.assertEqual(x,0)
		self.assertEqual(y,0)
		self.assertEqual(z,0)
		self.metasurface.set_unit_cell(100,-100,50)
		x, y, z = self.metasurface.get_unit_cell()
		self.assertEqual(x,100)
		self.assertEqual(y,-100)
		self.assertEqual(z,50)

	def test_particle(self):
		particle = self.metasurface.particles[0]
		self.assertEqual(particle.r_p,400/4)
		particle.ei = 9
		self.metasurface.set_particles(particle)
		self.assertEqual(self.metasurface.particles[0].ei,9)

	def test_set_alpha(self):
		self.metasurface.k = 0.01
		self.metasurface.set_alpha()
		alp = np.eye(6)
		self.metasurface.set_alpha_user_defined(alp)
		self.metasurface.alp_uc[0,0] = 1
