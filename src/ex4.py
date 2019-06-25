
# EX 3: plot a function

from ngsolve import *

ngsglobals.msg_level = 1

comm = mpi_world
if comm.rank in [1,2]:
    sub_comm = comm.SubComm([1,2])
    if sub_comm.rank == 0:
        from netgen.geom2d import unit_square
        mesh = unit_square.GenerateMesh(maxh=0.1)
        if sub_comm.size > 1:
            mesh.Distribute(sub_comm)
    else:
        from netgen.meshing import Mesh as NGMesh
        mesh = NGMesh.Receive(sub_comm)
    mesh = Mesh(mesh)
    print("rank", comm.rank, ", sub-comm rank", sub_comm.rank, "local NV =", mesh.nv, ", and NE =", mesh.ne)
    V = H1(mesh, order=1, dirichlet=".*")
    print("rank", comm.rank, ", sub-comm rank", sub_comm.rank, "has", V.ndof, "of", V.ndofglobal, "dofs!")
else:
    print("rank", comm.rank, "is bored, give me some work!")
