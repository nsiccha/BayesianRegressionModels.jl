# What BRM's module gives src/nlme_svi.jl and the files it includes, for the
# standalone arithmetic test modules that include those sources without loading
# BRM (nlme_svi_unit_curvature, nlme_svi_units, nlme_svi_weighted_iteration,
# nlme_svi_coupling_gradient). Include it inside such a module before
# src/nlme_svi.jl: the collapsed sparse globals (src/nlme_svi_collapsed.jl) extend
# Serialization and use the sparse factor and selected inversion, and the
# stationarity checks read MCMCDiagnosticTools' split-Rhat.
include(joinpath(@__DIR__, "fitting_packages.jl"))
import Serialization, MCMCDiagnosticTools, SparseArrays, SelectedInversion
# The standalone module includes the optimizer without loading BRM's extension.
_brm_rank_ess_rhat(x) = MCMCDiagnosticTools.ess_rhat(x; kind=:rank)
using SparseArrays: SparseMatrixCSC, sparse, spzeros, spdiagm, nonzeros, rowvals, nzrange,
    blockdiag
using SelectedInversion: selinv_diag, selinv, selinv_extract
