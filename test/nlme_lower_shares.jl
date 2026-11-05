# Run in the NLMEEstimation environment documented in test/README.md.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_shares_fixture.jl"))
include(joinpath(@__DIR__, "nlme_unit_checks.jl"))
using NLMEEstimation, Random, LogDensityProblems, LinearAlgebra

lower_panel(data) = @brm data begin
    sigma ~ Exponential(0.5)
    mu ~ 1 + z + (1 | region) + (1 | district)
    rows = groups(dy => cells)
    mi(x)::shares(tot, rows) ~ truncated(LogNormal(mu, sigma); lower=lo)
    lx = log(x)
    m ~ 1 + lx + (1 | region)
    w ~ Normal(m, 0.5)
end

@stestset "bounded shares retain NLME density, ownership and protocol" begin
    data = shares_panel(; nregion=2, ndistrict=1, nmuni=3, nyear=2)
    data = merge(data, (; lo=[ismissing(x) ? 2. : 1. for x in data.x]))
    rk = RKBRMI(lower_panel(data); completions=:noncentered)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, SHARES_NESTED...)
    @test !any(==(:unclaimed), m.view.role)
    family = only(k for (k, f) in enumerate(m.view.families) if get(f.block, :shares, false))
    @test first(only(NLMEEstimation.fixed_omega(m))) == family
    check_lockstep(m, Xoshiro(5))
    check_unit_problems(m, Xoshiro(6))
    start = brm_nlme_start(m)
    # Each cell's structural support may only contain one group's shares.
    names = BRM._rk_layout_coordinate_names(rk)
    c = only(rk.plan.completions)
    owned, next = Vector{Int}[], 0
    previous = 0
    for stop in rk.plan.columns[c.totals.ends]
        count = stop - previous - 1
        push!(owned, [findfirst(==(Symbol("x.z_shares.", next+j)), names) for j in 1:count])
        previous, next = stop, next + count
    end
    shared = reduce(vcat, owned)
    for cell in m.cell_query
        footprint = BRM._rk_nlme_cell_footprint_query(cell.query)
        for positions in cell.positions, p in positions
            read = intersect(BRM._rk_nlme_cell_footprint(footprint, m.unit, p), shared)
            @test isempty(read) || any(o -> issubset(read, o), owned)
        end
    end
    @test NLMEEstimation.check_protocol(m, start.theta, start.sigma, start.omega)
end
