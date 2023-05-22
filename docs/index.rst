.. cemd metasurf documentation master file, created by
   sphinx-quickstart on Sat May 20 17:46:14 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cemd metasurf's documentation!
=========================================

Welcome on the documentation for the coupled electric and magnetic dipoles (cemd) model for calculating optical properties of periodic metasurfaces. On this website, you will find a bit on the theory behind library, as well as the description of the classes and functions implemented, along with informative examples. The libray is under development and the current version can calculate reflection and transmissionn for lattices with one particle per unit cell, but it is planning to add new functionalities in future releases. For any comment or issue feel free to contact me at drabujetas@gmail.com.

Theory
-----------------

.. math::

   \begin{equation}
   \phi(\mathbf{r}) = G(\mathbf{r},\mathbf{r_i})\mathbf{\alpha_i} \phi(\mathbf{r_i})
   \end{equation}

Modules
-------

:doc:`source/api/my_metasurface`
    my_metasurface.py contais the initial classes for describing the basic properties of the metasurfaces (lattice parameters, arrangement of the unit cell and the polarizability of the particles) and for setting the Bloch wavevector at which the optical properties of the metasurface are going to be calculated.

:doc:`source/api/polarizability`
    The submodule polarizability have the functions for calculating the polarizability of spheres by using the Mie theory. It is possible that the particle clases (currently in the main module) will move to this submodules if more kind of particles are added. 

:doc:`source/api/depolarization_gf`
    The submodule depolarization_gf encloses the classes and function for calculating the depolarization Green function of the array.

:doc:`source/api/reflec_transm`
    In the submodule reflec_transm the classes and function for calculating reflectance and transmitance can be found.


Examples
--------

Below, different examples of how to use the library can be found.

:doc:`source/examples/example_rt`
    Example for calculating the reflectance at a constant angle of incidence for the metasurface used in [D.R. Abujetas, et. al., PRB, 102, 125411 (2020)].

.. Hidden TOCs

.. toctree::
   :maxdepth: 3
   :caption: Theory
   :hidden:

.. toctree::
   :maxdepth: 3
   :caption: Modules
   :hidden:

   source/api/my_metasurface
   source/api/polarizability
   source/api/depolarization_gf
   source/api/reflec_transm

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   source/examples/example_rt

.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

