# The matrix-free Laplace's checked conjugate-gradient solve (`_brm_checked_cg`,
# src/nlme_laplace_matrixfree.jl), loaded with that package (src/fitting_extensions.jl).
module BayesianRegressionModelsKrylovExt

import BayesianRegressionModels as BRM
import Krylov
using LinearAlgebra: norm

function BRM._brm_krylov_checked_cg(H, b, options; M, rtol, diagnostics)
    iszero(norm(b)) && return zeros(length(b))
    x, info = Krylov.cg_lanczos(H, collect(b); M,
        check_curvature=true, rtol,
        atol=0.0, itmax=options.solve_maxiter)
    residual = norm(H(x) .- b) / norm(b)
    if diagnostics !== nothing
        old = get(diagnostics, :linear_solves, (; attempts=0, max_iterations=0,
            max_relative_residual=0.0))
        diagnostics[:linear_solves] = (; attempts=old.attempts + 1,
            max_iterations=max(old.max_iterations, info.niter),
            max_relative_residual=max(old.max_relative_residual, residual))
    end
    info.solved && info.niter <= options.solve_maxiter &&
        residual <= options.solve_tolerance ||
        throw(BRM._BRMOperatorSolveFailure("conditional solve failed within " *
            "$(options.solve_maxiter) iterations (used $(info.niter), relative residual $residual; $(info.status))"))
    x
end

end
