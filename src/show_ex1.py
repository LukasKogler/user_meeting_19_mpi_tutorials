
# Load solution from EX 1

from ngsolve import *
from netgen.csg import unit_cube
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
V = H1(mesh, order=3, dirichlet=".*")
gfu = GridFunction(V)
gfu.Load('solution.sol', parallel=True)
Draw(gfu)
