# The sparse Laplace's inertia check (`_brm_spl_inertia`), loaded with that package
# (src/fitting_extensions.jl): MUMPS brings MPI, which no model build needs.
module BayesianRegressionModelsMUMPSExt

import BayesianRegressionModels as BRM
import MUMPS

# MUMPS's sequential backend still requires MPI initialization. Keep every
# MUMPS operation under one lock, and destroy its temporary native workspace
# before returning, including on errors. A chain retains only its KLU factor
# and selected-inverse workspace; deepcopy resets that whole cache.
const _BRM_INERTIA_LOCK = ReentrantLock()
function BRM._brm_mumps_inertia(K)
    lock(_BRM_INERTIA_LOCK) do
        MUMPS.MPI.Initialized() || MUMPS.MPI.Init(; threadlevel=:serialized)
        m = MUMPS.Mumps{Float64}(MUMPS.mumps_symmetric,
            MUMPS.get_icntl(det=true, nthreads=1, tree_parallelism=false),
            copy(MUMPS.default_cntl64); backend=MUMPS.Sequential())
        try
            MUMPS.set_icntl!(m, 13, 1; displaylevel=0)
            # Elimination can leave roundoff at an exactly singular Schur
            # complement. Reject numerical null pivots at MUMPS's default
            # scale-aware tolerance, even if its remaining inertia is right.
            MUMPS.set_icntl!(m, 24, 1; displaylevel=0)
            MUMPS.set_icntl!(m, 31, 1; displaylevel=0)
            MUMPS.associate_matrix!(m, K)
            MUMPS.set_job!(m, 4)
            MUMPS.invoke_mumps!(m)
            (; negative=m.infog[28] == 0 ? Int(m.infog[12]) : -1,
                logdet=log(abs(m.rinfog[12])) + m.infog[34] * log(2.0))
        catch err
            # A singular trial curvature is handled by Newton damping. Other
            # solver failures must remain visible, rather than looking like
            # a statistical rejection.
            err isa MUMPS.MUMPSException && m.infog[1] == -10 || rethrow()
            (; negative=-1, logdet=NaN)
        finally
            Base.finalize(m)
        end
    end
end

end
