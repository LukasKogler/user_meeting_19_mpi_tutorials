
ngsglobals.msg_level = 1


from netgen.geom2d import unit_square
mesh = unit_square.GenerateMesh(maxh=0.1)
mesh = Mesh(mesh)

V = H1(mesh, order=1, dirichlet=".*")

gfu = GridFunction(V)
gfu.Load("ex3.sol", parallel=True)
Draw(gfu)
