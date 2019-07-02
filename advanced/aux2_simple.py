
# First version of the auxiliary Preconditioner

from ngsolve import *
from netgen.csg import *
import auxstuff

comm = mpi_world

## The Mesh generation and definition of the main linear- and bilinear form
## is the same as before, 
# Note: alpha, beta are the coefficients for the  curl-curl and the l2 part respectively
mesh, HC, bfa, f, alpha, beta = auxstuff.SetUp(maxh=0.2, order=0)


# The Jacobi component of the Preconditioner
jac = Preconditioner(bfa, "local")


# per default, does nothing with MPI
with TaskManager():
    bfa.Assemble()
    f.Assemble()

    
# The gradient space component of the Preconditioner
    
## Gradient matrix and coresproding H1 space
G, H1s = HC.CreateGradient()

## G is a normal SparseMatrix, even when running with MPI !
## We have to wrap a ParalellMatrix around it ourselfs.
print(type(G))

## A ParallelMatrix consists of four ingredients:
##     - a local matrix
##     - ParallelDofs for the space of row- and col-vectors
##     - a PARALLEL_OP
## 
## The gradient matrix takes a sclar H1 function (Cumulated!)
## and returns a vector values HCurl function (Cumulated!)
## 
## This is different from bfa.mat, which takes a vector values HCurl (Cumulated!)
## and returns a dual vector (Distributed!)
##
## To reflect this, we wrap a ParallelMatrix of C2C (cumulated -> cumulated)
## type around it!

# print(help(ParallelMatrix))

if comm.size > 1:
    G = ParallelMatrix(G, row_pardofs = H1s.ParallelDofs(),
                       col_pardofs = HC.ParallelDofs(),
                       op=ParallelMatrix.C2C)

## The scalar preconditioner
us, vs = H1s.TnT()
h1s_blf = BilinearForm(H1s)
h1s_blf += beta * grad(us) * grad(vs) * dx
h1s_blf.Assemble()
pc_h1s = h1s_blf.mat.Inverse(H1s.FreeDofs())

# only inverse for now
pc_grad = G @ pc_h1s @ G.T



# The vector-H1 component of the Preconditioner
H1v = VectorH1(mesh, order=1, dirichlet="outer")

## First, we need the embedding E: HC -> H1v

# The simplest version is to set up a mixed mass matrix
# for H1v -> HC* and then solving with the mass matrix in HC
# for HC* -> HC
sigma, tau = HC.TnT()
hcmass = BilinearForm(HC)
hcmass += sigma * tau * dx
hcmass.Assemble()

uv, vv = H1v.TnT()
mixmass = BilinearForm(trialspace=H1v, testspace=HC)
mixmass += uv * tau * dx
mixmass.Assemble()

# This is a parallel inverse and can be used with the
# operator algebra without any issue
invtype = "sparsecholesky" if comm.size==1 else "mumps"
hcm_inv = hcmass.mat.Inverse(HC.FreeDofs(), inverse=invtype)

E = hcm_inv @ mixmass.mat
ET = mixmass.mat.T @ hcm_inv

## Next, we need a preconditioner in H1v
uv, vv = H1v.TnT()
h1v_blf = BilinearForm(H1v)
h1v_blf += alpha * InnerProduct(grad(uv), grad(vv)) * dx
h1v_blf.Assemble()
## for now, just the inverse
pc_h1v = h1v_blf.mat.Inverse(H1v.FreeDofs(), inverse=invtype)

pc_vec = E @ pc_h1v @ ET


# Now, define the Preconditioner
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
