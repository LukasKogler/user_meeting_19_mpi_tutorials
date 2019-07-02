
## Without auxiliary space PC
## Basically the same file as ngsolve/py_tutorials/cmagnet.py

from ngsolve import *
from netgen.csg import *

comm = mpi_world

geometry = CSGeometry()
box = OrthoBrick(Pnt(-1,-1,-1),Pnt(2,1,2)).bc("outer")

core = OrthoBrick(Pnt(0,-0.05,0),Pnt(0.8,0.05,1))- \
       OrthoBrick(Pnt(0.1,-1,0.1),Pnt(0.7,1,0.9))- \
       OrthoBrick(Pnt(0.5,-1,0.4),Pnt(1,1,0.6)).mat("core")

coil = (Cylinder(Pnt(0.05,0,0), Pnt(0.05,0,1), 0.3) - \
        Cylinder(Pnt(0.05,0,0), Pnt(0.05,0,1), 0.15)) * \
        OrthoBrick (Pnt(-1,-1,0.3),Pnt(1,1,0.7)).mat("coil")

geometry.Add ((box-core-coil).mat("air"))
geometry.Add (core)
geometry.Add (coil)

geometry.Draw()

mesh = Mesh(geometry.GenerateMesh(maxh=0.3))
ngsglobals.msg_level = 0
mesh.Curve(3)
ngsglobals.msg_level = 1

Draw(mesh)


HC = HCurl(mesh, order=0, dirichlet="outer")

mur = { "core" : 1000, "coil" : 1, "air" : 1 }
mu0 = 1.257e-6
nu_coef = [ 1/(mu0*mur[mat]) for mat in mesh.GetMaterials() ]
nu = CoefficientFunction(nu_coef)
alpha = nu
beta = 1e-6 * nu
sigma, tau = HC.TnT()

a = BilinearForm(HC)
a += alpha * curl(sigma) * curl(tau) * dx
a += beta * sigma * tau * dx
a.Assemble()

f = LinearForm(HC)
f += SymbolicLFI(CoefficientFunction((y,0.05-x,0)) * tau, definedon=mesh.Materials("coil"))
f.Assemble()

gfu = GridFunction(HC)

t = comm.WTime()
gfu.vec.data = a.mat.Inverse(HC.FreeDofs()) * f.vec
t = comm.WTime() - t

if comm.rank == 0:
    print(' ----------- ')
    print('ndof H-Curl space: ', HC.ndofglobal)
    print('t solve', t)
    print('dofs / (sec * np) ', HC.ndofglobal / (t * max(comm.size-1, 1)) )
    print(' ----------- ')

Draw (gfu.Deriv(), mesh, "B-field", draw_surf=False)
