
# EX 3: ParallelDofs

from ngsolve import *

ngsglobals.msg_level = 1

comm = mpi_world

if comm.rank == 0:
    from netgen.geom2d import unit_square
    mesh = unit_square.GenerateMesh(maxh=0.1)
    if comm.size > 1:
        mesh.Distribute(comm)
else:
    from netgen.meshing import Mesh as NGMesh
    mesh = NGMesh.Receive(comm)
mesh = Mesh(mesh)

V = H1(mesh, order=1, dirichlet=".*")

pardofs = V.ParallelDofs()
if comm.rank==1:
    for k in range(V.ndof):
        print("rank", comm.rank, "shares DOF", k, "with", list(pardofs.Dof2Proc(k)))

gfu = GridFunction(V)
for k in range(V.ndof):
    gfu.vec[k] = len(pardofs.Dof2Proc(k))
gfu.Save("ex3.sol", parallel=True)

# gfu.vec.SetParallelStatus(PARALLEL_STATUS.DISTRIBUTED)
# gfu.vec[:] = 1
# gfu.vec.Cumulate()
# gfu.Save("ex3.sol", parallel=True)
