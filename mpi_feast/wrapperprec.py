import ngsolve as ng


class WrapperPrec(ng.BaseMatrix):
    def __init__ (self, A):
        super(WrapperPrec, self).__init__()
        self.A = A
        
        self.v = A.CreateRowVector()
        self.vout = A.CreateColVector()

    def Mult (self, x, y):
        # Get the real part of the input vector x into self.v
        self.v.SetParallelStatus(x.GetParallelStatus())
        for k in range(len(self.v)):
            self.v[k] = x[k].real


        # Apply A to v.
        self.vout.data = self.A * self.v

        # Set the real part of the computation into the output y.
        y.SetParallelStatus(self.vout.GetParallelStatus())
        for k in range(len(self.vout)):
            y[k] = self.vout[k]


        # Get the imaginary part of the input vector x into self.v

        self.v.SetParallelStatus(x.GetParallelStatus())
        for k in range(len(self.v)):
            self.v[k] = x[k].imag

        # Apply A to v.
        self.vout.data = self.A * self.v

        # Add the real part of the computation into the output y.
        
        # Cumulate or distribute y according to the parallel status of vout.
        par_status = self.vout.GetParallelStatus()

        if par_status == ng.PARALLEL_STATUS.DISTRIBUTED:
            y.Distribute()
        else:
            y.Cumulate()

        for k in range(len(self.vout)):
            y[k] += self.vout[k]

    def Height (self):
        return self.A.height

    def Width (self):
        return self.A.height


#if __name__ == '__main__':
#    real_pc = petsc.PETSc2NGsPrecond(real_mat, some_more_options)
#    wrapper_pc = WrapperPrec(real_pc)
