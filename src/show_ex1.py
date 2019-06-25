
# Load solution from EX 1

from ngsolve import *
from netgen.csg import unit_cube
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.3))
V = H1(mesh, order=0, dirichlet=".*")
gfu = GridFunction(V)
gfu.Load('solution.sol', parallel=True)
Draw(gfu)
