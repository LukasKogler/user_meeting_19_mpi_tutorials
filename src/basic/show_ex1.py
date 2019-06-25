
# Load solution from EX 1

from ngsolve import *
from netgen.geom2d import unit_square
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
V = H1(mesh, order=3, dirichlet=".*")
gfu = GridFunction(V)
gfu.Load('ex1.sol', parallel=True)
Draw(gfu)
