# Run a repository test file with every `brm_svi_fit` call at an explicit
# `unit_reoptimize = K`, its assertions, thresholds and references unchanged
# (todo 12maq21; snag brm-svi-fit-defa-cfc1bb5e's gate ran the same way). A file
# that passes `unit_reoptimize` itself is refused, not overridden. Never read a red
# K = 1 comparison as a pass: a historical negative control written for the k = 0
# crawl can fail because the crawl is gone.
#
#     julia --project=test research/svi_reoptimize/kmode_test.jl K test/<file>.jl [filter...]
using BayesianRegressionModels
const K = parse(Int, popfirst!(ARGS))
path = abspath(popfirst!(ARGS))
source = read(path, String)
occursin("unit_reoptimize", source) && error("$path passes unit_reoptimize itself")
kfit(args...; kwargs...) = brm_svi_fit(args...; unit_reoptimize=K, kwargs...)
source = replace(source, "brm_svi_fit(" => "kfit(")
println("KMODE K=$K file=$path filters=$ARGS")
include_string(Main, source, path)
