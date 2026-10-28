# test/rk_cdar.jl — `cdar(step; by=g, cor=C)` on RKBRMI and its NLME reading.
#
# RKBRMI lowers the per-group stationary AR(1) walk with SBBRMI's law: the
# marginal SD `sigma` and persistence `rho`, standardized innovations `eta` on
# the group × step grid, `delta[:, 1] = sigma L eta[:, 1]` and
# `delta[:, w] = rho delta[:, w-1] + sigma sqrt(1 - rho^2) L eta[:, w]`.
# In `brm_nlme_view`, a walk whose groups are the units, or nest in them (each
# group lies in one unit), holds each group's innovations as unit draws of the
# subject owning the group, one family fixed at the identity; `sigma` and
# `rho` stay shared. A `cor=` coupling groups of different units, or groups
# that are not units, keeps the walk shared as a whole.
#
# Covered: RK = SB density and gradient at transported coordinates (`cor = I`,
# a correlated `cor`, stated and known priors), the walk against an
# independent recurrence, printed-source replay, the NLME roles, families and
# subject blocks, lockstep and unit-problem densities against the full RK
# density, and per-unit Laplace and SVI on the nested model.
#
# Run: julia --project=test test/rk_cdar.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random, StanBlocks
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))
include(joinpath(@__DIR__, "nlme_unit_checks.jl"))

const BS = StanBlocks.BridgeStan
const CDAR_CACHE = joinpath(tempdir(), "brm-rk-cdar")

include(joinpath(@__DIR__, "rk_cdar_fixture.jl"))

instantiate(sb) = begin
    code = StanBlocks.stan_code(sb.model)
    mkpath(CDAR_CACHE)
    StanBlocks.stan_instantiate(sb.model; path=joinpath(CDAR_CACHE, string(hash(code)) * ".stan"))
end

@stestset "RKBRMI: the same law as SBBRMI at transported coordinates" begin
    for model in (walk_model, correlated_model, across_model, known_model, region_model)
        brmi = model()
        rk = check_rk_source_roundtrip(RKBRMI(brmi))
        problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        stan = instantiate(sb)
        stan_names = BS.param_unc_names(stan.model)
        transport = brm_coordinate_transport(rk, sb, stan_names)
        @test length(transport) == length(coordinate_names(rk.model.layout)) ==
            length(stan_names)
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))))
            stan_u = brm_rk_point_to_stan(transport, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_value, stan_gradient = BS.log_density_gradient!(stan.model, stan_u,
                zeros(length(stan_u)); propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=1e-9 rtol=1e-12
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=1e-8 rtol=1e-10
        end
    end
end

@stestset "RKBRMI: the walk equals the stationary recurrence" begin
    rk = RKBRMI(bare_model())
    names = coordinate_names(rk.model.layout)
    @test names[1:2] == [Symbol("cdar_mu_year.sigma"), Symbol("cdar_mu_year.rho")]
    @test names[3:end] == [Symbol("cdar_mu_year.eta.", g, ".", w) for w in 1:4 for g in 1:6]
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec; have=(:unconstrained, keys(bound)...),
        want=(:mu,), bound)
    L = cholesky(Symmetric(WALK_DF.C2)).L
    for u in (collect(range(-0.6, 0.7; length=length(names))), 0.3 .* randn(Xoshiro(8), length(names)))
        sigma, rho = exp(u[1]), 1 / (1 + exp(-u[2]))
        eta = reshape(u[3:end], 6, 4)
        delta = zeros(6, 4)
        delta[:, 1] = sigma * (L * eta[:, 1])
        for w in 2:4
            delta[:, w] = rho * delta[:, w - 1] + sigma * sqrt(1 - rho^2) * (L * eta[:, w])
        end
        expected = [delta[parse(Int, k[2:end]), t] for (k, t) in CELLS]
        value = Base.invokelatest(query, u)
        @test (value isa Tuple ? only(value) : value) ≈ expected atol=1e-12 rtol=1e-12
    end
end

@stestset "RKBRMI: a replay of the walk onto new rows is not built yet" begin
    rk = RKBRMI(walk_model())
    # todo: frozen RK replay and forecasts of `cdar` (SBBRMI has both).
    @test_broken (reprocess(rk, WALK_DF); true)
end

@stestset "NLME view: a walk nested in the units holds each group's innovations" begin
    rk = RKBRMI(walk_model())
    v = brm_nlme_view(rk; NESTED...)
    @test !any(==(:unclaimed), v.role)
    @test BRM._brm_nlme_ragged(v)
    k = cdar_family(v)
    family = v.families[k]
    @test family.size == 4 && family.fixed == Matrix{Float64}(I, 4, 4)
    @test isempty(family.scales) && isempty(family.factor)
    @test family.block.group === :kreis
    shared = v.coordinates[v.role .=== :shared_latent]
    @test sort(shared) == [Symbol("cdar_mu_year.rho"), Symbol("cdar_mu_year.sigma")]
    @test count(==(:subject_effect), v.role) == 3 + 6 * 4
    for (i, level) in enumerate(v.levels)
        districts = sort([k for k in KREIS if REGION_OF[k] == level])
        @test v.subject_families[i] == [1, k, k]
        margins = [b for b in v.subject_margins[i] if hasproperty(b.block, :cdar)]
        @test [b.level for b in margins] == repeat(districts; inner=4)
        @test [b.coefficient for b in margins] ==
            repeat([Symbol("cdar(year)[", t, "]") for t in 1:4], 2)
        walk = v.coordinates[v.subject_coordinates[i][2:end]]
        @test walk == [Symbol("cdar_mu_year.eta.", findfirst(==(d), KREIS), ".", w)
            for d in districts for w in 1:4]
    end
    m = brm_nlme_model(rk; ad_backend=NLME_AD, NESTED...)
    check_lockstep(m, Xoshiro(1))
    check_unit_problems(m, Xoshiro(2))
end

@stestset "NLME view: a cor= within the units keeps the innovations unit-owned" begin
    rk = RKBRMI(correlated_model())
    v = brm_nlme_view(rk; NESTED...)
    @test cdar_family(v) !== nothing
    m = brm_nlme_model(rk; ad_backend=NLME_AD, NESTED...)
    check_lockstep(m, Xoshiro(3))
    check_unit_problems(m, Xoshiro(4))
end

@stestset "NLME view: a cor= across the units, or shared_groups, keeps the walk shared" begin
    rk = RKBRMI(across_model())
    v = brm_nlme_view(rk; NESTED...)
    @test cdar_family(v) === nothing
    walk = findall(n -> startswith(String(n), "cdar_mu_year."), v.coordinates)
    @test length(walk) == 2 + 24 && all(i -> v.role[i] === :shared_latent, walk)
    # Without `nested_groups` the districts name no subject: shared too, and
    # so when `shared_groups` names them.
    @test cdar_family(brm_nlme_view(RKBRMI(walk_model()))) === nothing
    v = brm_nlme_view(RKBRMI(walk_model()); subject_group=:region, shared_groups=(:kreis,))
    @test cdar_family(v) === nothing
    walk = findall(n -> startswith(String(n), "cdar_mu_year."), v.coordinates)
    @test length(walk) == 2 + 24 && all(i -> v.role[i] === :shared_latent, walk)
    # refused: a grouping is owned by one subject or crossed with the subjects,
    # never both (the `shared_groups` contract, snag crossed-grouping-39d39fcc).
    @test_throws "name it in one" brm_nlme_view(RKBRMI(walk_model());
        subject_group=:region, nested_groups=(kreis=:region,), shared_groups=(:kreis,))
end

@stestset "NLME view: a walk grouped by the units" begin
    rk = RKBRMI(region_model())
    v = brm_nlme_view(rk)
    @test !BRM._brm_nlme_ragged(v)
    k = cdar_family(v)
    @test v.families[k].size == 4
    @test all(==([1, k]), v.subject_families)
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    check_lockstep(m, Xoshiro(5))
    check_unit_problems(m, Xoshiro(6))
end

@stestset "per-unit Laplace and SVI take the nested walk" begin
    rk = RKBRMI(walk_model())
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, NESTED...)
    @test sum(length, p.model.view.subject_coordinates) == 3 + 24
    x = zeros(LogDensityProblems.dimension(p))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
    @test isfinite(value) && all(isfinite, gradient)
    fit = brm_svi_fit(rk; rng=Xoshiro(7), ad_backend=NLME_AD, batch_size=2,
        max_iterations=200, ndraws=20, min_windows=1, NESTED...)
    @test fit.coordinates == p.model.view.coordinates
    @test size(fit.draws, 2) == 20 && all(isfinite, fit.draws)
end

println("rk_cdar.jl: all testsets passed")
