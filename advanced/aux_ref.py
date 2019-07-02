
# Using 

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

pre = Preconditioner(bfa, "petsc_pc_hypre_ams")
# pre = Preconditioner(bfa, "hypre_ams")

with TaskManager():
    bfa.Assemble()
    f.Assemble()

gfu = GridFunction(HC)

t = comm.WTime()
solvers.CG(mat=bfa.mat, pre=pre, sol=gfu.vec, rhs=f.vec, tol=1e-6,
           printrates=comm.rank==0)
t = comm.WTime() - t

if comm.rank == 0:
    print(' ----------- ')
    print('ndof H-Curl space: ', HC.ndofglobal)
    print('t solve', t)
    print('dofs / (sec * np) ', HC.ndofglobal / (t * max(comm.size-1, 1)) )
    print(' ----------- ')

