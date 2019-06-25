
from ngsolve import *

ngsglobals.msg_level = 0

from netgen.geom2d import unit_square
mesh = unit_square.GenerateMesh(maxh=0.1)
ngsmesh = Mesh(mesh)
mesh = Mesh(mesh)

V = L2(mesh, order=0)

gfu = GridFunction(V)
gfu.Load("ex2.sol", parallel=True)
Draw(gfu)
