
# EX 4: time-DG with MPI and periodic boundary

from ngsolve import *

comm = mpi_world

# We can create and then distribute a periodic mesh as usual !
# (This duplicates surface elements on the periodic boundary)

if comm.rank==0:
    from netgen.geom2d import unit_square
    from netgen.geom2d import *
    geo = SplineGeometry()
    pnts = [ (0,0), (1,0), (1,1), (0,1) ]
    pnums = [geo.AppendPoint(*p) for p in pnts]
    lbot = geo.Append ( ["line", pnums[0], pnums[1]],bc="bot")
    # geo.Append ( ["line", pnums[3], pnums[2]], leftdomain=0, rightdomain=1, bc="top")
    geo.Append ( ["line", pnums[3], pnums[2]], leftdomain=0, rightdomain=1, copy=lbot, bc="top")
    lright = geo.Append ( ["line", pnums[1], pnums[2]], bc="right")
    geo.Append ( ["line", pnums[0], pnums[3]], leftdomain=0, rightdomain=1, copy=lright, bc="left")
    ngmesh = geo.GenerateMesh(maxh=0.05)
    if comm.size>1:
        ngmesh.Distribute(comm)
else:
    from netgen.meshing import Mesh as NGMesh
    ngmesh = NGMesh.Receive(comm)
mesh = Mesh(ngmesh)

V = L2(mesh, order=4)

u,v = V.TnT()

# b = CoefficientFunction( (0.2 + 0.5 * (y-0.5), 0.5 * (0.5-x)) )
b = CoefficientFunction( (0.3 + (y-0.5), (0.5-x)) )
bn = b*specialcf.normal(2)

a = BilinearForm(V)
a += SymbolicBFI (-u * b*grad(v))
a += SymbolicBFI ( bn*IfPos(bn, u, u.Other()) * (v-v.Other()), VOL, skeleton=True)
a += SymbolicBFI ( bn*IfPos(bn, u, 0) * v, BND, skeleton=True)

u = GridFunction(V)
u.Set(exp (-40 * ( (x-0.7)*(x-0.7) + (y-0.7)*(y-0.7) )))

w = u.vec.CreateVector()

t = 0
tau = 5e-4
tend = 2
count = 0
out_interval = 0.02 // tau


Draw(u, name="sol")

if comm.size > 1:
    import os
    output_path = os.path.dirname(os.path.realpath(__file__)) + "/solutions"
    if comm.rank == 0 and not os.path.exists(output_path):
        os.mkdir(output_path)
    comm.Barrier()
    while t < tend:
        if comm.rank == 0:
            print("\rt =", t, ", # of files =", int(count/out_interval), end=' ')
        a.Apply (u.vec, w)
        V.SolveM (rho=CoefficientFunction(1), vec=w)
        u.vec.data -= tau * w
        t += tau
        if count % out_interval == 0:
            u.Save(output_path + '/u_' + str(int(count/out_interval)) + '.sol', parallel=True)
        count = count+1;
    comm.Barrier()
    if comm.rank == 0:
        print('')
else:
    import time
    while t < tend:
        if count % out_interval == 0:
            print("\rload fileno =", int(count/out_interval), end=' ')
            u.Load('solutions/u_' + str(int(count/out_interval)) + '.sol', parallel=True)
        t += tau
        count = count+1;
        Redraw(blocking=True)
        time.sleep(0.0005)
    print('')
