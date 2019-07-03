"""
Script that uses the FEAST eigensolver to find the first three
eigenvalues of the negative laplacian on the unit square with
zero dirichlet boundary conditions.
"""

import ngsolve as ng
import numpy as np
import mpi4py.MPI as MPI
from mpi_feast_utils import feast


if __name__ == '__main__':
    # Get the world communicator.
    ng_comm = ng.MPI_Init()

    # Also get the world communitor for mpi4py.
    m4p_comm = MPI.COMM_WORLD

    r = 20.0            # The contour radius.
    c = 30.0 + 0.0j     # The contour center.
    N = 8               # Number of quadrature points on the contour.
    maxh = 0.2          # The maximum initial mesh size.
    nref = 2            # The number of times to refine the initial mesh.
    order = 2           # The order of the finite element space.
    niters = 50         # The number of FEAST iterations.
    reltol = 1.e-13     # The relative stopping tolerance for the eigenvalues.


    eigenvalues, eigenvectors = \
        feast(ng_comm, m4p_comm, r=r, c=c, N=N, maxh=maxh, nref=nref,
              order=order, niters=niters, reltol=reltol)

    if ng_comm.rank == 0:
        print('Approximate eigenvalues:', eigenvalues)

        exact_eigenvalues = np.array([2*np.pi**2, 5*np.pi**2, 5*np.pi**2])
        print('Exact eigenvalues:', exact_eigenvalues)
