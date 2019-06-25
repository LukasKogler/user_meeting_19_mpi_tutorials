
# EX 1: super simple poisson

from ngsolve import *

ngsglobals.msg_level = 1

# get handle to the MPI communicator
comm = mpi_world
print("Hello from rank", comm.rank, "of", comm.size)

if comm.rank == 0:
    # the master-proc generates mesh
    from netgen.geom2d import unit_square
    mesh = unit_square.GenerateMesh(maxh=0.1)
    ngsmesh = Mesh(mesh)
    print("global NV =", ngsmesh.nv, ", and NE =", ngsmesh.ne)
    if comm.size > 1:
        mesh.Distribute(comm)
else:
    # the others wait to receive their part of the mesh
    # from the master
    from netgen.meshing import Mesh as NGMesh
    mesh = NGMesh.Receive(comm)
mesh = Mesh(mesh)

print("rank", comm.rank, ", local NV =", mesh.nv, ", and NE =", mesh.ne)

# build H1-FESpace as usual
V = H1(mesh, order=3, dirichlet=".*")
print("rank", comm.rank, "has", V.ndof, "of", V.ndofglobal, "dofs!")

u,v = V.TnT()

# RHS does not change either!
f = LinearForm (V)
f += 32 * (y*(1-y)+x*(1-x)) * v * dx
f.Assemble()

# neither does the BLF!
a = BilinearForm (V)
a += grad(u) * grad(v) * dx

gfu = GridFunction(V)

## CG + Jacobi Preconditioner
c = Preconditioner(a, type="local")
a.Assemble()


solvers.CG(mat=a.mat, pre=c.mat, sol=gfu.vec, rhs=f.vec, maxsteps=100,
           tol=1e-6, printrates=comm.rank==0) # !

gfu.Save("solution.sol", parallel=True)
