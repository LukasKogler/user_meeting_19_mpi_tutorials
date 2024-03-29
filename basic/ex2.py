
# EX 2: Visualize Mesh Distribution

from ngsolve import *

ngsglobals.msg_level = 0

comm  = mpi_world

if comm.rank == 0:
    from netgen.geom2d import unit_square
    mesh = unit_square.GenerateMesh(maxh=0.1)
    ngsmesh = Mesh(mesh)
    if comm.size > 1:
        mesh.Distribute(comm)
else:
    from netgen.meshing import Mesh as NGMesh
    mesh = NGMesh.Receive(comm)
mesh = Mesh(mesh)


# To visualize the partitioning of the mesh, we take
# a lowest order L2 function and set it to our rank

V = L2(mesh, order=0)
gfu = GridFunction(V)
if comm.size>1:
    gfu.Set(comm.rank)
    gfu.Save("ex2.sol", parallel=True)
else:
    gfu.Load("ex2.sol", parallel=True)
    Draw(gfu)

## Exercise: Why does this not work so well with H1?
