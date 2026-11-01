# Run with the repository's test environment and a compute token.
# Reuse the authored fixture and its inexpensive structural checks.
empty!(ARGS); push!(ARGS, "selections stay compact")
include(joinpath(@__DIR__, "../../test/laplace_factors.jl"))

function measure_assembly(n)
    rk = spl_sum_fixture(n)
    packed = length(BRM._rk_layout_coordinate_names(rk))
    u = zeros(packed)
    f = BRM._brm_spl_prepare_factors(rk, :y, u, 1, 1:packed, Int[])
    f = BRM._brm_spl_factor_operators(f, u, NLME_AD)
    BRM._brm_spl_factor_curvature(f, u, packed, 0)
    term = first(first(only(f.cells).terms))
    BRM._rk_spl_factor_block(term.ad, u, term.sel, 1; term=term.t)
    function terms()
        for _ in 1:100
            BRM._rk_spl_factor_block(term.ad, u, term.sel, 1; term=term.t)
        end
    end
    terms()
    singles = [@timed terms() for _ in 1:3]
    runs = [@timed BRM._brm_spl_factor_curvature(f, u, packed, 0) for _ in 1:3]
    a = first(runs).value
    result = (; n, packed, local_entries=nnz(a.D), sum_entries=nnz(a.B),
        outer_entries=nnz(a.W), seconds=minimum(x.time for x in runs),
        bytes=minimum(x.bytes for x in runs),
        term_seconds=minimum(x.time for x in singles)/100,
        term_bytes=minimum(x.bytes for x in singles)/100)
    println(result)
    result
end

results = map(measure_assembly, (32, 512, 4096))
@test results[2].term_bytes == results[3].term_bytes
@test results[3].bytes < 10results[2].bytes
