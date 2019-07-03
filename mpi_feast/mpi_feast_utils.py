"""
Script that contains utility functions for running
the FEAST eigensolver for the negative laplacian
operator with zero dirichlet boundary conditions on
the unit square.
"""

from ngsolve.la import InnerProduct
from math import sqrt
from ngsolve import Projector, Norm
from ngsolve.ngstd import Timer
import ngsolve

import mpi4py.MPI as MPI
import netgen
import netgen.meshing
from netgen.geom2d import unit_square
import ngsolve as ng
import ngsolve.solvers as solvers
import numpy as np
import scipy.linalg as spla
import ngs_petsc as petsc
from wrapperprec import WrapperPrec
import sys
from pathlib import Path


def MyGMRes(A, b, pre=None, freedofs=None, x=None, maxsteps = 100, tol = 1e-7, innerproduct=None,
          callback=None, restart=None, startiteration=0, printrates=True):
    """
    Important Note: This version of GMRes only differs from NGSolve's
    current version of GMRes in that the assert statement for freedofs is
    now

        assert freedofs is not None

    Otherwise, everything else remains the same.
    """
    if not innerproduct:
        innerproduct = lambda x,y: y.InnerProduct(x, conjugate=True)
        norm = ngsolve.Norm
    else:
        norm = lambda x: ngsolve.sqrt(innerproduct(x,x).real)
    # is_complex = isinstance(b.FV(), ngsolve.bla.FlatVectorC)
    is_complex = b.is_complex
    if not pre:
        assert freedofs is not None
        pre = ngsolve.Projector(freedofs, True)
    n = len(b)
    m = maxsteps
    if not x:
        x = b.CreateVector()
        x[:] = 0

    if callback:
        xstart = x.CreateVector()
        xstart.data = x
    else:
        xstart = None
    sn = ngsolve.Vector(m, is_complex)
    cs = ngsolve.Vector(m, is_complex)
    sn[:] = 0
    cs[:] = 0

    r = b.CreateVector()
    tmp = b.CreateVector()
    tmp.data = b - A * x
    r.data = pre * tmp

    Q = []
    H = []
    Q.append(b.CreateVector())
    r_norm = norm(r)
    if abs(r_norm) < tol:
        return x
    Q[0].data = 1./r_norm * r
    beta = ngsolve.Vector(m+1, is_complex)
    beta[:] = 0
    beta[0] = r_norm

    def arnoldi(A,Q,k):
        q = b.CreateVector()
        tmp.data = A * Q[k]
        q.data = pre * tmp
        h = ngsolve.Vector(m+1, is_complex)
        h[:] = 0
        for i in range(k+1):
            h[i] = innerproduct(Q[i],q)
            q.data += (-1)* h[i] * Q[i]
        h[k+1] = norm(q)
        if abs(h[k+1]) < 1e-12:
            return h, None
        q *= 1./h[k+1].real
        return h, q

    def givens_rotation(v1,v2):
        if v2 == 0:
            return 1,0
        elif v1 == 0:
            return 0,v2/abs(v2)
        else:
            t = ngsolve.sqrt((v1.conjugate()*v1+v2.conjugate()*v2).real)
            cs = abs(v1)/t
            sn = v1/abs(v1) * v2.conjugate()/t
            return cs,sn

    def apply_givens_rotation(h, cs, sn, k):
        for i in range(k):
            temp = cs[i] * h[i] + sn[i] * h[i+1]
            h[i+1] = -sn[i].conjugate() * h[i] + cs[i].conjugate() * h[i+1]
            h[i] = temp
        cs[k], sn[k] = givens_rotation(h[k], h[k+1])
        h[k] = cs[k] * h[k] + sn[k] * h[k+1]
        h[k+1] = 0

    def calcSolution(k):
        mat = ngsolve.Matrix(k+1,k+1, is_complex)
        for i in range(k+1):
            mat[:,i] = H[i][:k+1]
        rs = ngsolve.Vector(k+1, is_complex)
        rs[:] = beta[:k+1]
        y = mat.I * rs
        if xstart:
            x.data = xstart
        for i in range(k+1):
            x.data += y[i] * Q[i]

    for k in range(m):
        startiteration += 1
        h,q = arnoldi(A,Q,k)
        H.append(h)
        if q is None:
            break
        Q.append(q)
        apply_givens_rotation(h, cs, sn, k)
        beta[k+1] = -sn[k].conjugate() * beta[k]
        beta[k] = cs[k] * beta[k]
        error = abs(beta[k+1])
        if printrates:
            print("Step", startiteration, ", error = ", error)
        if callback:
            calcSolution(k)
            callback(x)
        if error < tol:
            break
        if restart and k+1 == restart and not (restart == maxsteps):
            calcSolution(k)
            del Q
            return MyGMRes(A, b, freedofs=freedofs, pre=pre, x=x, maxsteps=maxsteps-restart, callback=callback,
                           tol=tol, innerproduct=innerproduct,
                           restart=restart, startiteration=startiteration, printrates=printrates)
    calcSolution(k)
    return x


def spaces_and_forms(mesh, order=1):
    # Create a distributed finite element space.
    dirichlet = 'top|right|bottom|left'
    X = ng.H1(mesh, order=order, dirichlet=dirichlet,
              complex=True)

    # Create the test and trial functions.
    u, v = X.TnT()

    # Create the bilinear form for the left-hand-side.
    a = ng.BilinearForm(X)
    a += ng.SymbolicBFI(ng.grad(u) * ng.grad(v))
    a.Assemble()

    # Create the second needed bilinear form as is needed for FEAST.
    b = ng.BilinearForm(X)
    b += ng.SymbolicBFI(u * v)
    b.Assemble()   

    return X, a, b


def systems_and_preconditioners(mesh, X, z, order=1):
    # Create a real analogue of our vector space for the wrapper preconditioner.
    dirichlet = 'top|right|bottom|left'
    X_real = ng.H1(mesh, order=order, dirichlet=dirichlet,
                   complex=False)

    # Test and trial functions for the original space.
    u, v = X.TnT()

    # Real trial and test functions.
    ur, vr = X_real.TnT()

    # Create a real analogue of the bilinear form a.
    a_real = ng.BilinearForm(X_real)
    a_real += ng.SymbolicBFI(ng.grad(ur) * ng.grad(vr))
    a_real.Assemble()

    # Initialize petsc prior to conswtructing preconditioners.
    petsc.Initialize()

    # Create a bilinear form and a preconditioner for each z.
    zbas = []
    precs = []
    for k in range(len(z)):
        #  Create a bilinear form for the given z-value.
        zba = ng.BilinearForm(X)
        zba += ng.SymbolicBFI(z[k] * u * v - ng.grad(u) * ng.grad(v))

        # Create a preconditioner for the given z-value.
        mat_convert = petsc.PETScMatrix(a_real.mat, freedofs=X_real.FreeDofs())
        real_pc = petsc.PETSc2NGsPrecond(
            mat=mat_convert, 
            name="real_pc",
            petsc_options = {"pc_type" : "gamg"}
        )

        prec = WrapperPrec(real_pc)

        # Assemble the given bilinear form.
        zba.Assemble()

        # Tack the bilinear forms and preconditioners onto their respective lists.
        zbas += [zba]
        precs += [prec]

    return zbas, precs


def create_mesh(ng_comm, maxh=0.2, nrefinements=0):
    sequential = ng_comm.size == 1
    output_directory = 'outputs'

    if sequential:
        path_to_mesh = Path(output_directory + '/mesh.vol')

        if path_to_mesh.exists():
            print('Sequential: Loading mesh from {0}.'.format(output_directory))
            mesh = ng.Mesh(output_directory + '/mesh.vol')
        else:
            print('Sequential: Creating and saving mesh to {0}.'.format(output_directory))
            ngmesh = unit_square.GenerateMesh(maxh=maxh)

            for k in range(nrefinements):
                ngmesh.Refine()

            ngmesh.Save(output_directory + '/mesh.vol')

            mesh = ng.Mesh(ngmesh)
    else:
        if ng_comm.rank == 0:
            print('Rank {0}: Loading mesh from {1}.'.format(ng_comm.rank, output_directory))

        mesh = ng.Mesh(output_directory + '/mesh.vol')

    return mesh


def compute_weights_and_points(r=1.0, c=0.0+0.0j, N=4):
    """
    Method that computes the quadrature weights and nodes for a circle
    of radius r centered in the complex plane about c.
    """
    theta = np.linspace(0, 2 * np.pi, num=N, endpoint=False)

    # Shift the points so that a quadrature point does not coincide
    # (hopefully) with a desired eigenvalue.
    dtheta = (2 * np.pi) / N
    theta += 0.5 * dtheta
    z = r * np.exp(1j * theta) + c
    w = r * np.exp(1j * theta) / N

    return w, z


def inital_span(X):
    # Coefficient funtions in the span.
    coeffs = (ng.CoefficientFunction([1]), ng.x, ng.y)

    # The span as a list.
    span = [ng.GridFunction(X) for k in range(len(coeffs))]

    # Set the values of each GridFunction.
    for k in range(len(coeffs)):
        span[k].Set(coeffs[k])

    return span


def rayleigh(A, B, q):
    """
    Return (A q, q) and (B q, q)
    """

    dims = (len(q), len(q))
    qAq = np.zeros(dims, dtype=complex)
    qBq = np.zeros(dims, dtype=complex)
    tmp = q[0].vec.CreateVector()

    for k in range(dims[0]):
        for l in range(k, dims[1]):
            tmp.data = A * q[l].vec.data
            qAq[k, l] = ng.InnerProduct(tmp, q[k].vec)
            qAq[l, k] = np.conjugate(qAq[k, l])

            tmp.data = B * q[l].vec.data
            qBq[k, l] = ng.InnerProduct(tmp, q[k].vec)
            qBq[l, k] = np.conjugate(qBq[k, l])

    return (qAq, qBq)


def linear_combo(v, W):
    """
    Method that multiplies the list v containing the span of our vectors
    to the matrix W that solved the eigenproblem

        vAv*W = vBv*W*E,

    where E is the diagonal matrix of eigenvalues.
    """

    # Create an array of new eigenfunctions.
    q = []
    for k in range(len(v)):
        q  += [ng.GridFunction(v[k].space)]
        q[k].Set(ng.CoefficientFunction(0))

    # Now set the values of each new grid function.
    for k in range(len(q)):
        for m in range(len(v)):
            q[k].vec.data += np.complex(W[m, k]) * v[m].vec

    return q


def apply_spectral_projector(ng_comm, X, zbas, precs, Bq, w, P_q0):

    # Create a scratch grid function for this computation.
    tmp = ng.GridFunction(X)

    for k in range(len(zbas)):
        for m in range(len(Bq)):
            tmp.Set(0.0 + 0.0j)

            # Run the conjugate gradient method. This step computes the
            # solution w = (zB - A)^{-1} B*f[m] for each of the m
            # right-hand-sides f[m].
            MyGMRes(
                A=zbas[k].mat,
                b=Bq[m],
                pre=precs[k],
                freedofs=None,
                x=tmp.vec,
                maxsteps=2000,
                tol=1e-14,
                innerproduct=None,
                restart=None,
                printrates=False
            )

            P_q0[m].vec.data += np.complex(w[k]) * tmp.vec


def mpi_feast_hermitian_step(ng_comm, m4p_comm, X, zbas, precs, Bq0, A, B, w, z, q0, P_q0):


    # Perform the application of the spectral projector.
    apply_spectral_projector(ng_comm, X, zbas, precs, Bq0, w, P_q0)

    # Solve the small Rayleigh Ritz problem.
    qAq, qBq = rayleigh(A, B, P_q0)
    dtype = qAq.dtype

    # Print out+\ the results of the rayleigh ritz procedure.
    print('qAq =\n', qAq)
    print('--------------------------------------\n')

    print('qBq =\n', qBq)
    print('--------------------------------------\n')
    ng_comm.Barrier()

    eigenvalues, eigenvectors = spla.eigh(qAq, qBq)

    # For fun, have the last rank print out its eigenvalues and eigenvectors.
    print('Rank', m4p_comm.Get_rank(), '\n',
          'eigenvalues =', eigenvalues, '\n',
          'eigenvectors =\n', eigenvectors)
    print('--------------------------------------\n')
    ng_comm.Barrier()

    qnew = linear_combo(P_q0, eigenvectors)

    return eigenvalues, qnew


def feast(ng_comm, m4p_comm, r=20.0, c=30.0+0.0j, N=4, maxh=0.2, nref=0,
          order=2, niters=1, reltol=1e-13):
    # Compute the quadrature weights and points.
    w, z = compute_weights_and_points(r=r, c=c, N=N)

    # Rank 0 will announce our quadrature points.
    if ng_comm.rank == 0:
        print('Computing shifted linear systems at z =')
        for k in range(len(z)):
            print('\tz[{0}] = {1}'.format(k, z[k]))

    # The ouputs directory containing the mesh (also the locations to which we
    # will dump the mpi gridfunctions).
    output_directory = 'outputs'

    # Generate the mesh.
    mesh = create_mesh(ng_comm, maxh=maxh, nrefinements=nref)

    # Get the spaces and forms for the Laplace eigenvalue problem.
    X, a, b = spaces_and_forms(mesh=mesh, order=order)

    # Get the shifted systems and preconditioners for feast.
    zbas, precs = systems_and_preconditioners(mesh, X, z, order=order)

    # The span as a list of grid functions.
    q0 = inital_span(X)

    # Create an array of grid functions that represent the application of the
    # spectral projector to each initial grid function in the initial span q0.
    P_q0 = [ng.GridFunction(X)
            for k in range(len(q0))]

    # Create a list of right-hand-side vectors multiplied by B := b.mat.
    Bq0 = [q0[k].vec.CreateVector() for k in range(len(q0))]

    # Array of eigenvalue arrays.
    ev_hist = []
    relerr_hist = []

    # Tht total number of iterations.

    for itr in range(niters):
        # Set the right-hand-sides at each iteration.
        for k in range(len(Bq0)):
            Bq0[k].data = b.mat * q0[k].vec

        # Run one step of feast.
        eigenvalues, q = mpi_feast_hermitian_step(ng_comm, m4p_comm, X, zbas,
                                                  precs, Bq0, a.mat, b.mat, w,
                                                  z, q0, P_q0)

        ev_hist += [eigenvalues]
        
        if itr > 0:
            relerr_hist += [
                np.abs((ev_hist[-1] - ev_hist[-2]) / ev_hist[-1])
            ]

        # Compute the relative errors at each step and have rank 0 report.
        if ng_comm.rank == 0:
            if itr > 0:
                print('Iteration {0}: Relative Error ='.format(itr), relerr_hist[-1])

        # Save the parallel solutions.
        #middlename = 'gridfunction_{0:02d}'.format(itr)
        middlename = 'gridfunction'
        save_funcs(q, ng_comm.size == 1, middlename=middlename)

        if itr > 0:
            if np.max(relerr_hist[-1]) < reltol:
                total_iterations = itr + 1

                break

        # Reset the initial guess.
        for k in range(len(q0)):
            q0[k].Set(q[k])

        # Zero out the intermediate coefficient functions.
        for k in range(len(P_q0)):
            P_q0[k].Set(ng.CoefficientFunction(0))

    if ng_comm.rank == 0:
        if len(relerr_hist) > 0:
            relerr_max = np.max(relerr_hist[-1])

            if relerr_max < reltol:
                print('FEAST converged with')
            else:
                total_iterations = niters
                print('Warning: FEAST may not have converged with')

            print('\tIterations: {0}'.format(total_iterations))
            print('\tMaximum Relative Error: {0}'.format(relerr_max))

    # Clean up petsc resources.
    petsc.Finalize()

    return eigenvalues, q


def save_funcs(q, sequential, middlename='gridfunction', output_directory='outputs'):
    filename = ''

    if sequential:
        filename += 'sequential_'
    else:
        filename += 'mpi_'

    filename += middlename + '_'
    filename += '{0:02d}'

    filename = output_directory + '/' + filename

    for k in range(len(q)):
        q[k].Save(filename.format(k), parallel=not sequential)
