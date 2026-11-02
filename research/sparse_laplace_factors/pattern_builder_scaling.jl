# Execute the complete production pattern method with a minimal input container,
# isolating all its retained index metadata from model lowering and AD setup.
# Each invocation measures one point; use /usr/bin/time -v for independent RSS.
using BayesianRegressionModels, SparseArrays, SHA
module PatternProbe
using BayesianRegressionModels, SparseArrays
const BRM = BayesianRegressionModels
Base.@kwdef mutable struct BRMSparseLaplaceProblem
    integrated::Vector{Int} = Int[]
    family::Vector{Int} = Int[]
    entry::Vector{Int} = Int[]
    rblock::Vector{Int} = Int[]
    rptr::Vector{Int} = Int[]
    rlevel::Vector{Int} = Int[]
    blocks::Vector{Any} = Any[]
    cells::Vector{Any} = Any[]
    tptr::Vector{Int} = Int[]
    tz::Vector{Int} = Int[]
    rloc::Vector{Int} = Int[]
    diagonal::Vector{Int} = Int[]
    prior_nz::Vector{Int} = Int[]
    prior_val::Vector{Float64} = Float64[]
    wcols::Vector{Vector{Int}} = Vector{Int}[]
    wλ::Vector{Float64} = Float64[]
    pattern_stats::NamedTuple = (;)
    pattern::SparseMatrixCSC{Float64,Int} = spzeros(0, 0)
    component::Vector{Int} = Int[]
    components::Vector{Vector{Int}} = Vector{Int}[]
    term_component::Vector{Int} = Int[]
end
const _BRMSumCellGroup = BRM._BRMSumCellGroup
const _brm_spl_pattern = BRM._brm_spl_pattern
const _brm_spl_nzindex = BRM._brm_spl_nzindex
const _brm_spl_identity = BRM._brm_spl_identity
const _brm_nlme_error = BRM._brm_nlme_error
const _brm_lap_width = BRM._brm_lap_width
_brm_spl_nterms(p) = length(p.family) + sum(g -> length(g.cells), p.cells; init=0)
const source_path = joinpath(pkgdir(BRM), "src", "nlme_laplace_sparse.jl")
source = read(source_path, String)
signature = "function _brm_spl_pattern!(p::BRMSparseLaplaceProblem)"
body = only(split(source, signature)[2:end])
method = signature * first(split(body, "# One term's C rows"))
include_string(@__MODULE__, method, "production_pattern_method.jl")
function fixture(width, count, shape)
    n = shape == "repeated" ? width : width + count - 1
    cells = [(; lz=collect(shape == "repeated" ? (1:width) : (i:(i+width-1))), term=i)
        for i in 1:count]
    BRMSparseLaplaceProblem(; integrated=collect(1:n), cells=Any[(; cells)])
end
end
width, cells = parse.(Int, ARGS[1:2])
shape = length(ARGS) >= 3 ? ARGS[3] : "repeated"
shape in ("repeated", "overlap") || error("shape must be repeated or overlap")
PatternProbe._brm_spl_pattern!(PatternProbe.fixture(4, 2, shape))
p = PatternProbe.fixture(width, cells, shape)
GC.gc()
measurement = @timed PatternProbe._brm_spl_pattern!(p)
@assert length(p.tz) == width*cells
@assert length(p.tptr) == cells+1
@assert length(p.components) == 1
@assert all(iszero, nonzeros(p.pattern))
@assert :tnz ∉ fieldnames(BayesianRegressionModels.BRMSparseLaplaceProblem)
@assert :nptr ∉ fieldnames(BayesianRegressionModels.BRMSparseLaplaceProblem)
s = p.pattern_stats
metadata = (p.tptr,p.tz,p.rloc,p.diagonal,p.prior_nz,p.prior_val,p.wcols,p.wλ,
    p.component,p.components,p.term_component,p.pattern_stats)
println("shape\twidth\tcells\tinput_indices\tdistinct_footprints\tpair_visits\tnnz\tcell_input_bytes\tpattern_bytes\tindex_metadata_bytes\ttotal_fixture_bytes\tretained_pair_map_entries\tallocated_bytes\tseconds\tprocess_maxrss_bytes\tpattern_source_sha256")
println(join((shape,width,cells,s.input_indices,s.distinct_footprints,s.pair_visits,
    nnz(p.pattern),Base.summarysize(p.cells),Base.summarysize(p.pattern),
    Base.summarysize(metadata),Base.summarysize(p),0,measurement.bytes,
    measurement.time,Sys.maxrss(),bytes2hex(sha256(read(PatternProbe.source_path)))), '\t'))
