# The matrix-free Laplace's fixed reference precision (`_brm_fixed_reference`,
# src/nlme_laplace_preconditioner.jl), loaded with that package
# (src/fitting_extensions.jl). The result holds plain sparse factors only.
module BayesianRegressionModelsLimitedLDLFactorizationsExt

import BayesianRegressionModels as BRM
import LimitedLDLFactorizations
import SparseArrays
using SparseArrays: spdiagm
using LinearAlgebra: UnitLowerTriangular, diag

function BRM._brm_lldl_fixed_reference(H, memory; ridge)
    # Do not expand reduction cliques B' W B. Its diagonal can still improve
    # the reference; the complete reduction remains in every H action.
    d = H.diagonal .+ H.shift
    Q = H.D + spdiagm(0 => d .- diag(H.D) .+ ridge .* abs.(d))
    F = LimitedLDLFactorizations.lldl(Q; memory)
    all(isfinite, F.D) && all(!iszero, F.D) ||
        throw(BRM._BRMOperatorSolveFailure("nonfinite or singular fixed conditional preconditioner"))
    L = UnitLowerTriangular(copy(F.L))
    BRM._BRMFixedReference(L, L', copy(F.P), 1 ./ sqrt.(abs.(F.D)),
        sum(log, abs.(F.D)), SparseArrays.nnz(F.L), Float64(F.α))
end

end
