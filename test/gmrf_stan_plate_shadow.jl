# Full lexical-shadow acceptance on StanBlocks' explicit-local cell contract.
# The independent priors and frozen replay must agree with the same oracle
# even when caller data contain a same-named correlation matrix.
include("gmrf_stan_plate_correlation.jl")
gmrf_stan_plate_correlations(; shadow=true)
