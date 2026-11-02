# Run each point in a separate process under /usr/bin/time -v for peak RSS:
# julia --project=test research/sparse_laplace_factors/pattern_scaling.jl 128 400 repeated
# Input incidence, pair visits, retained storage, and allocation are separate.
using BayesianRegressionModels, SparseArrays
const BRM = BayesianRegressionModels
width, cells = parse.(Int, ARGS[1:2])
shape = length(ARGS) >= 3 ? ARGS[3] : "repeated"
shape in ("repeated", "overlap") || error("shape must be repeated or overlap")
footprints = shape == "repeated" ? [collect(1:width) for _ in 1:cells] :
    [collect(i:(i + width - 1)) for i in 1:cells]
n = shape == "repeated" ? width : width + cells - 1
edges = Tuple{Int,Int}[]
BRM._brm_spl_pattern(footprints, n, edges) # compile before allocation measurement
GC.gc()
measurement = @timed BRM._brm_spl_pattern(footprints, n, edges)
result = measurement.value
println("shape\twidth\tcells\tinput_indices\tdistinct_footprints\tdistinct_indices\tpair_visits\tnnz\tpattern_bytes\tstatistics_bytes\tallocated_bytes\tseconds\tprocess_maxrss_bytes")
s = result.stats
println(join((shape, width, cells, s.input_indices, s.distinct_footprints,
    s.distinct_indices, s.pair_visits, s.retained_nnz, Base.summarysize(result.pattern),
    Base.summarysize(s), measurement.bytes, measurement.time, Sys.maxrss()), '\t'))
