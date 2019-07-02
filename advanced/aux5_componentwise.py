
# Applying component-wise solver for the vector component


from ngsolve import *
from netgen.csg import *
import auxstuff
import ngs_petsc

use_ngs_amg = True
try:
    import ngs_amg
except:
    use_ngs_amg = False
    
comm = mpi_world

mesh, HC, bfa, f, alpha, beta = auxstuff.SetUp(maxh=0.2, order=0)

jac = Preconditioner(bfa, "local")

with TaskManager():
    bfa.Assemble()
    f.Assemble()

G, H1s = HC.CreateGradient()
if comm.size > 1:
    G = ParallelMatrix(G, row_pardofs = H1s.ParallelDofs(),
                       col_pardofs = HC.ParallelDofs(),
                       op=ParallelMatrix.C2C)

## The scalar preconditioner
us, vs = H1s.TnT()
h1s_blf = BilinearForm(H1s)
h1s_blf += beta * grad(us) * grad(vs) * dx

# Using NGsAMG
if use_ngs_amg:
    pc_h1s = Preconditioner(h1s_blf, "ngs_amg.h1_scal")

h1s_blf.Assemble()

# Using GAMG algebraic multigrid from PETSc
if not use_ngs_amg:
    pm_scal = ngs_petsc.PETScMatrix(h1s_blf.mat, H1s.FreeDofs())
    pc_h1s = ngs_petsc.PETSc2NGsPrecond(pm_scal, petsc_options = {"pc_type" : "gamg"})

pc_grad = G @ pc_h1s @ G.T

H1v = VectorH1(mesh, order=1, dirichlet="outer")

## On monday, we saw fast embedding via Dual shapes
## (see fast_embed.py)
from fast_embed import FastEmbed
E = FastEmbed(HC, H1v)

# Again, this is a C2C operation, because it takes a vector-valued
# H1 function (!) nad returns an HCurl function (!)
if mpi_world.size > 1:
    E = ParallelMatrix(E, row_pardofs = H1v.ParallelDofs(),
                        col_pardofs = HC.ParallelDofs(),
                        op = ParallelMatrix.C2C)

# This time, for the vector component, instead of one big preconditioner
# for all components of the H1 function, be apply a preconditioner to
# each component seperately

H1vs = H1v.components[0]
uv, vv = H1vs.TnT()
h1v_blf = BilinearForm(H1vs)
h1v_blf += alpha * InnerProduct(grad(uv), grad(vv)) * dx

if use_ngs_amg:
    pc_h1v_comp = Preconditioner(h1v_blf, "ngs_amg.h1_scal")

h1v_blf.Assemble()

if not use_ngs_amg:
    pm_vec = ngs_petsc.PETScMatrix(h1v_blf.mat, H1v.FreeDofs())
    pc_h1v_comp = ngs_petsc.PETSc2NGsPrecond(pm_vec, petsc_options = {"pc_type" : "gamg"})


# We derive from BaseMatrix
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
        # apply the small preconditioner to every component of the large input vector seperately
        for l in range(3):
            self.x.local_vec.data = x.local_vec[ l * self.comp_nds : (l+1) * self.comp_nds]
            self.x.SetParallelStatus(x.GetParallelStatus())
            self.y.data = scal * self.pc * self.x
            self.y.Cumulate()
            y.local_vec[ IntRange(l * self.comp_nds, (l+1) * self.comp_nds)] += self.y.local_vec

pc_h1v = ComponentWisePC(pc_h1v_comp, H1v)

pc_vec = E @ pc_h1v @ E.T

aux_pre = pc_vec + pc_grad + jac

gfu = GridFunction(HC)

t = comm.WTime()
solvers.CG(mat=bfa.mat, pre=aux_pre, sol=gfu.vec, rhs=f.vec, tol=1e-6,
           printrates=comm.rank==0)
t = comm.WTime() - t

if comm.rank == 0:
    print(' ----------- ')
    print('ndof H-Curl space: ', HC.ndofglobal)
    print('t solve', t)
    print('dofs / (sec * np) ', HC.ndofglobal / (t * max(comm.size-1, 1)) )
    print(' ----------- ')
    
