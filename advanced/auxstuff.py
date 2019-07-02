
from ngsolve import *
from netgen.csg import *


def SetUp (maxh = 0.3, order = 0, condense = False, jump = 1, nref = 0):

    def make_geo():
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
        return geometry

    geo = make_geo()
    comm = mpi_world

    if comm.rank==0:
        # ngsglobals.msg_level = 0
        ngmesh = geo.GenerateMesh(maxh=maxh)
        # ngsglobals.msg_level = 1
        if comm.size > 1:
            ngmesh.Distribute(comm)
    else:
        from netgen.meshing import Mesh as NGMesh
        ngmesh = NGMesh.Receive(comm)
        ngmesh.SetGeometry(geo)
    for k in range(nref):
        ngmesh.Refine()

    mesh = Mesh(ngmesh)
        
    ngsglobals.msg_level = 0
    mesh.Curve(3)
    ngsglobals.msg_level = 1

    HC = HCurl(mesh, order=order, dirichlet="outer")

    mur = { "core" : jump, "coil" : 1, "air" : 1 }
    mu0 = 1.257e-6
    nu_coef = [ 1/(mu0*mur[mat]) for mat in mesh.GetMaterials() ]
    nu = CoefficientFunction(nu_coef)
    alpha = nu
    beta = 1e-6 * nu
    sigma, tau = HC.TnT()

    a = BilinearForm(HC)
    a += alpha * curl(sigma) * curl(tau) * dx
    a += beta * sigma * tau * dx

    f = LinearForm(HC)
    f += SymbolicLFI(CoefficientFunction((y,0.05-x,0)) * tau, definedon=mesh.Materials("coil"))

    return mesh, HC, a, f, alpha, beta
