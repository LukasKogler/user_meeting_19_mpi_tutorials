
# Using fast Embedding via dual shapes

from ngsolve import *
from netgen.csg import *
import auxstuff
import ngs_petsc

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
# pc_h1s = Preconditioner(h1s_blf, "ngs_amg.h1_scal")

h1s_blf.Assemble()

# Using GAMG algebraic multigrid from PETSc
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

uv, vv = H1v.TnT()
h1v_blf = BilinearForm(H1v)
h1v_blf += alpha * InnerProduct(grad(uv), grad(vv)) * dx
h1v_blf.Assemble()

# Again, simply GAMG
pm_vec = ngs_petsc.PETScMatrix(h1v_blf.mat, H1v.FreeDofs())
pc_h1v = ngs_petsc.PETSc2NGsPrecond(pm_vec, petsc_options = {"pc_type" : "gamg"})

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
    
