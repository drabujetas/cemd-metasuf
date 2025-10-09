
import numpy as np	
from . import field_functions as nff

class NearF(object):

    def get_near_field_kxky(self, xyz, my_bloch = None, ind_ini = -1, pol_tm = 1/np.sqrt(2), pol_te = 1j/np.sqrt(2)):
        """
        Calculates the near field at the given positions and Bloch wavevectors.

        :param xyz:
        :type xyz: numpy.ndarray 
        :param my_bloch: The object with the information of the wavevectos at which gb is calculated.
        :type my_bloch: classes.BlochWavevector 
        :param ind_ini: Which value of the stored gb use to calculate the near field. Only used is "my_bloch = None". By default use the last one.
        :type ind_ini: Int
        :param pol_tm: TM amplitude incident wave.
        :type pol_tm: complex
        :param pol_te: TE amplitude incident wave.
        :type pol_te: complex 
        """
        n_field = xyz.shape[0]
        x_array = xyz[:,0]
        y_array = xyz[:,1]
        z_array = xyz[:,2]
        e_field = np.zeros((n_field, 3), dtype = 'complex_' )
        h_field = np.zeros((n_field, 3), dtype = 'complex_' )
        if type(my_bloch) != type(None):
            self.calc_gb_kxky(my_bloch, append_k_gb = True)

        self.gb_kxky = self.array_k_gb[ind_ini,3:].reshape(6,6)
        self.k, self.kx, self.ky = self.array_k_gb[ind_ini,0:3].real
        self.set_alpha()
        for i in range(n_field):
            e_i, h_i = nff.calc_near_field_kxky(self, pol_tm, pol_te, x_array[i],y_array[i],z_array[i])
            e_field[i,:] = e_i[:,0]
            h_field[i,:] = h_i[:,0]
        return e_field, h_field
