# The sparse Laplace's augmented-system factor (`_BRMAugmentedFactor.F`,
# src/nlme_laplace_factors.jl): KLU's factorization, its solves and its scaled
# pivots, loaded on first use (src/lazy_dependencies.jl).
module BayesianRegressionModelsKLUExt

import BayesianRegressionModels as BRM
import KLU
using LinearAlgebra: diag

BRM._brm_klu(K) = KLU.klu(K)
BRM._brm_klu_solve(F, rhs) = F \ rhs
# KLU's scaled identity is L*U+F = Diagonal(1 ./ Rs)*K[p,q].
BRM._brm_klu_scaled_pivots(F) = (; d=diag(F.U), p=F.p, Rs=F.Rs)

end
