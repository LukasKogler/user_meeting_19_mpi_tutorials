"""
A small script that loads the results of sequential and parallel FEAST
using NGSolve.
"""

import ngsolve as ng

if __name__ == '__main__':
    try:
        # Load the mesh from file.
        mesh = ng.Mesh('outputs/mesh.vol')

        # Create our H1 finite element space.
        order = 2
        dirichlet = 'top|right|bottom|left'
        X = ng.H1(mesh, order=order, dirichlet=dirichlet,
                  complex=True)

        # Create an index array for the gridfunctions.
        idx = list(range(0, 3))

        gf_sequential = []
        gf_mpi = []

        middlename = 'gridfunction'

        for k in range(len(idx)):
            # Create two grid functions: One for the sequential solve, and one
            # for the MPI solve.
            sequential_name = 'sequential_' + middlename + '_{0:02d}'
            gf_sequential += [ng.GridFunction(X, name=sequential_name.format(idx[k]))]

            mpi_name = 'mpi_' + middlename + '_{0:02d}'
            gf_mpi += [ng.GridFunction(X, name=mpi_name.format(idx[k]))]

            # Load the grid functions from file.
            sequential_filename = 'outputs/' + sequential_name
            gf_sequential[-1].Load(sequential_filename.format(idx[k]), parallel=False)

            mpi_filename = 'outputs/' + mpi_name
            gf_mpi[-1].Load(mpi_filename.format(idx[k]), parallel=True)

        # Draw the grid functions.
        for k in range(len(idx)):
            ng.Draw(gf_sequential[k])
            ng.Draw(gf_mpi[k])

    except Exception as e:
        print('An error occurred:', e)
