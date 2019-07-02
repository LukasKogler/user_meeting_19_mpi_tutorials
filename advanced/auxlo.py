from ngsolve import *
from netgen.csg import *
import time
import ngs_amg 

maxh = 0.5
diri = "outer"
order = 1
condense = True

def MakeGeometry():
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


import ngs_petsc as petsc
petsc.Initialize()
comm = mpi_world

ngsglobals.msg_level = 0
geom = MakeGeometry()
if False:
    if comm.rank==0:
        ngmesh = geom.GenerateMesh(maxh=maxh)
        ngmesh.Save('cmag.vol')
        ngmesh.Distribute(comm)
    else:
        from netgen.meshing import Mesh as NGMesh
        ngmesh = NGMesh.Receive(comm)
        ngmesh.SetGeometry(geom)
    ngmesh.Refine()
    mesh = Mesh(ngmesh)
    ngsglobals.msg_level = 0
    # mesh.Curve(3)
    ngsglobals.msg_level = 3
else:
    mesh = Mesh('cmag.vol', comm)
    ngsglobals.msg_level = 0
    mesh.ngmesh.SetGeometry(geom)
    mesh.Curve(3)
    ngsglobals.msg_level = 3

# paje_size = 50 * 1024 * 1024 if comm.rank in [0,1] else 0
paje_size = 0
### Set up the HCurl problem
with TaskManager(pajetrace = paje_size):

    HC = HCurl(mesh, order=order, dirichlet=diri)

    mur = { "core" : 1000, "coil" : 1, "air" : 1 }
    mu0 = 1.257e-6
    nu_coef = [ 1/(mu0*mur[mat]) for mat in mesh.GetMaterials() ]
    nu = CoefficientFunction(nu_coef)
    alpha = nu
    beta = 1e-6 * nu
    sigma, tau = HC.TnT()
    a = BilinearForm(HC, condense=condense)
    a += alpha * curl(sigma) * curl(tau) * dx
    a += beta * sigma * tau * dx
    jac = Preconditioner(a, "local")
    a.Assemble()

    f = LinearForm(HC)
    f += SymbolicLFI(CoefficientFunction((y,0.05-x,0)) * tau, definedon=mesh.Materials("coil"))
    f.Assemble()

    ### Set up the Gradient Space portion of the Preconditioner

    ## Gradient matrix and coresproding H1 space
    # loHC = HC.lospace # ( <- does not give us an HCurl space)
    loHC = HCurl(mesh, order=0, dirichlet=diri)
    G, H1s = loHC.CreateGradient()
    hcemb = Embedding(HC.ndof, IntRange(0, loHC.ndof))
    G = hcemb @ G
    if comm.size > 1:
        G = ParallelMatrix(G, row_pardofs = H1s.ParallelDofs(), col_pardofs = HC.ParallelDofs(),
                           op=ParallelMatrix.C2C)

    us, vs = H1s.TnT()
    h1s_blf = BilinearForm(H1s, condense=condense)
    h1s_blf += beta * grad(us) * grad(vs) * dx
    if False: #order <= 1:
        pc_h1s = Preconditioner(h1s_blf, "ngs_amg.h1_scal", ngs_amg_log_level = 2)
    else:
        pc_h1s = Preconditioner(h1s_blf, "bddc", coarsetype="ngs_amg.h1_scal", ngs_amg_log_level = 2,
                                # ngs_amg_lower = 0, ngs_amg_upper = mesh.nv,
                                ngs_amg_test = True)
        # pc_h1s = Preconditioner(h1s_blf, "bddc", coarsetype="petsc_pc", petsc_pc_pc_type = "ml" )
        #pc_h1s = Preconditioner(h1s_blf, "bddc")
    h1s_blf.Assemble()

    ngsglobals.msg_level = 1
    pc_h1s.Test()

    # pm = petsc.PETScMatrix(h1s_blf.mat, H1s.FreeDofs())
    # pc_h1s = petsc.KSP(pm, petsc_options={"pc_type":"gamg"})
    # pc_h1s = petsc.PETSc2NGsPrecond(pm, petsc_options={"pc_type":"gamg"})
        

    ## Putting the gradient range preconditioner together
    pcgrad = G @ pc_h1s @ G.T

    
    ### Set up the Vector-H1 portion of the Preconditioner

    H1v = VectorH1(mesh, order=1, dirichlet=diri)

    ## H1->HCurl embedding
    if True:
        from fast_embed import FastEmbed
        E = FastEmbed(V_GOAL=HC, V_ORIGIN=H1v)
        ET = E.T
    else:
        hcmass = BilinearForm(HC)
        hcmass += sigma * tau * dx
        hcmass.Assemble()

        uv, vv = H1v.TnT()
        mixmass = BilinearForm(trialspace=H1v, testspace=HC)
        mixmass += uv * tau * dx
        mixmass.Assemble()

        hcm_inv = hcmass.mat.Inverse(HC.FreeDofs(), inverse="sparsecholesky" if comm.size==1 else "mumps")

        E = hcm_inv @ mixmass.mat
        ET = mixmass.mat.T @ hcm_inv

    ## Vector-H1 space preconditioner (component wise!)
    H1vs = H1v.components[0]
    uv, vv = H1vs.TnT()
    h1v_blf = BilinearForm(H1vs, condense=condense)
    h1v_blf += alpha * InnerProduct(grad(uv), grad(vv)) * dx
    # h1v_blf += alpha * InnerProduct(uv, vv) * dx
    if True: # order <= 1:
        pc_h1v_comp = Preconditioner(h1v_blf, "ngs_amg.h1_scal", ngs_amg_log_level = 1)
    else:
        pc_h1v_comp = Preconditioner(h1v_blf, "bddc", coarsetype="ngs_amg.h1_scal", ngs_amg_log_level = 1,
                                     ngs_amg_test = True)

    h1v_blf.Assemble()

    ngsglobals.msg_level = 1
    pc_h1v_comp.Test()

    ## Putting the vector-h1 preconditioner together
    class ComponentWisePC (BaseMatrix):
        def __init__(self, pc, V):
            super(ComponentWisePC, self).__init__()
            self.gfvec = GridFunction(V)
            self.gfscal = GridFunction(V.components[0])
            self.pc = pc
            self.x = self.gfscal.vec.CreateVector()
            self.y = self.gfscal.vec.CreateVector()
            self.comp_nds = V.components[0].ndof
            
        def Height(self):
            return len(self.gfvec.vec)
        def Width(self):
            return len(self.gfvec.vec)
        def CreateRowVector(self):
            return self.gfvec.vec.CreateVector()
        def CreateColVector(self):
            return self.gfvec.vec.CreateVector()

        def MultAdd(self, scal, x, y):
            y.Cumulate()
            for l in range(3):
                self.x.local_vec.data = x.local_vec[ l * self.comp_nds : (l+1) * self.comp_nds]
                self.x.SetParallelStatus(x.GetParallelStatus())
                self.y.data = scal * self.pc * self.x
                self.y.Cumulate()
                y.local_vec[ IntRange(l * self.comp_nds, (l+1) * self.comp_nds)] += self.y.local_vec
                
    pc_h1v = ComponentWisePC(pc_h1v_comp, H1v)
    pcvec = E @ pc_h1v @ ET

    gfu = GridFunction(HC)

    ### The smoother component
    pcsmo = jac.mat

    # hcmat = petsc.PETScMatrix(a.mat, HC.FreeDofs(condense))
    # pcsmo = petsc.PETSc2NGsPrecond(hcmat, "pcsmo", petsc_options = {"pc_type" : "sor"})

    # the full preconditioner
    class MultiplicativePC (BaseMatrix):
        def __init__(self, pcs, A):
            super(MultiplicativePC, self).__init__()
            self.pcs = pcs
            self.res = self.pcs[0].CreateColVector()
            self.sol = self.pcs[0].CreateColVector()
            self.A = A
            
        def Height(self):
            return self.pcs[0].height
        def Width(self):
            return self.pcs[0].width
        def CreateRowVector(self):
            return self.pcs[0].CreateRowVector()
        def CreateColVector(self):
            return self.pcs[0].CreateColVector()

        def MultAdd(self, scal, rhs, sol):
            self.sol.data = self.pcs[0] * rhs
            for pc in self.pcs[1:]:
                self.res.data = rhs - self.A * self.sol
                self.sol.data += pc * self.res
            sol.data += scal * self.sol

    pc = pcvec + pcgrad + jac
    # pc = MultiplicativePC(pcs = [pcgrad, pcvec, jac], A = a.mat)

    # pam = petsc.PETScMatrix(a.mat, HC.FreeDofs(condense))
    pam = petsc.FlatPETScMatrix(a.mat, HC.FreeDofs(condense))
    ksp = petsc.KSP(mat=pam, name="aux_ksp",
                    petsc_options = {"ksp_rtol" : 1e-6,
                                     "ksp_norm_type" : "preconditioned",
                                     "ksp_view_eigenvalues" : "",
                                     "ksp_atol" : 1e-50,
                                     "ksp_type" : "cg",
                                     "ksp_max_it" : 500,
                                     "ksp_monitor" : "",
                                     "ksp_converged_reason" : ""},
                    finalize=False)
    pcpc = petsc.ConvertNGsPrecond(pc, mat=pam, name="ngs_side_aux_pc")
    ksp.SetPC(pcpc)
    ksp.Finalize()

    comm.Barrier()
    t1 = -time.time()
    gfu.vec.data = ksp * f.vec
    comm.Barrier()
    t1 = t1 + time.time()

    comm.Barrier()
    t2 = -time.time()
    solvers.CG(mat=a.mat, pre=pc, rhs=f.vec, sol=gfu.vec, tol=1e-6, maxsteps=500, printrates=mpi_world.rank==0)
    comm.Barrier()
    t2 = t2 + time.time()


    

    if comm.rank == 0:
        print(' ----------- ')
        print('ndof H-Curl space: ', HC.ndofglobal)
        print('low order ndof H-Curl space: ', HC.lospace.ndofglobal)
        print(' --- KSP --- ')
        print('t solve', t1)
        print('dofs / (sec * np) ', HC.ndofglobal / (t1 * max(comm.size-1, 1)) )
        print(' --- NGs-CG --- ')
        print('t solve', t2)
        print('dofs / (sec * np) ', HC.ndofglobal / (t2 * max(comm.size-1, 1)) )
        print(' ----------- ')


    ex_sol = False
    if ex_sol:
        err = f.vec.CreateVector()
        exsol = f.vec.CreateVector()

        exsol.data = a.mat.Inverse(HC.FreeDofs(condense)) * f.vec
        err.data = exsol - gfu.vec
        nerr = Norm(err)

        if comm.rank==0:
            print('err ', nerr)
