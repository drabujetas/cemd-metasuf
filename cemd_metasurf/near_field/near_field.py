
import numpy as np	
from . import field_functions as nff

class NearField(object):

    def get_near_field_kxky(self, xyz, my_bloch, pol_tm = 1/np.sqrt(2), pol_te = 1j/np.sqrt(2)):
        """
        Calculates the near field at the given positions and Bloch wavevectors.

        :param xyz:
        :type xyz: numpy.ndarray 
        :param my_bloch: The object with the information of the wavevectos at which gb is calculated.
        :type my_bloch: classes.BlochWavevector 
        """
        n_field = xyz.shape[0]
        x_array = xyz[:,0]
        y_array = xyz[:,0]
        z_array = xyz[:,0]
        e_field = np.zeros((n_field, 3), dtype = 'complex_' )
        h_field = np.zeros((n_field, 3), dtype = 'complex_' )
        for i in range(n_field):
            e_i, h_i = nff.calc_near_field_kxky(self,my_bloch, pol_tm, pol_te, x_array[i],y_array[i],z_array[i])
            e_field[i,:] = e_i
            h_field[i,:] = h_i
        return e_field, h_field
