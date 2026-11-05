# Retained preparation must save actual setup calls, preserve the marginal,
# and give every target independent state, including strict term subsets.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, Random, LinearAlgebra

mutable struct CountingLaplaceAD
    rows::Int
    cells::Int
    factors::Int
end
CountingLaplaceAD() = CountingLaplaceAD(0, 0, 0)
counts(ad) = (ad.rows, ad.cells, ad.factors)
BRM._brm_enzyme_mode(mode, ::CountingLaplaceAD) = BRM._brm_enzyme_mode(mode, NLME_AD)
function BRM._rk_slap_kernels(rk::RKBRMI, ports::Tuple, effects::Tuple,
        omega::Vector{Int}, u::Vector{Float64}, ad::CountingLaplaceAD)
    ad.rows += 1
    BRM._rk_slap_kernels(rk, ports, effects, omega, u, NLME_AD)
end
function BRM._rk_spl_cell_ad(rk::RKBRMI, obs::Tuple, u::Vector{Float64},
        indices::Vector{Int}, ad::CountingLaplaceAD, K::Int; kwargs...)
    ad.cells += 1
    BRM._rk_spl_cell_ad(rk, obs, u, indices, NLME_AD, K; kwargs...)
end
function BRM._rk_spl_factor_ad(q, part, u, indices, ad::CountingLaplaceAD; kwargs...)
    ad.factors += 1
    BRM._rk_spl_factor_ad(q, part, u, indices, NLME_AD; kwargs...)
end

function check_plan_reuse(rk, groups; partial=false, cellkind=:none)
    ad = CountingLaplaceAD()
    plan = brm_laplace_plan(rk; ad_backend=ad, integrate=partial ? (first(groups),) : groups,
        draws=12, ess_floor=1.1)
    @test count(t -> t.admitted, plan.terms) == (partial ? 1 : length(groups))
    prepared = counts(ad)
    @test prepared[1] == 1
    cellkind === :ordinary && @test prepared[2] > 0
    cellkind === :nested && @test prepared[2] + prepared[3] > 0
    cellkind === :sum && @test prepared[3] > 0
    elapsed = @timed brm_laplace_problem(rk; ad_backend=ad, integrate=plan,
        tol=1e-20, maxiter=70)
    p = elapsed.value
    q = brm_laplace_problem(rk; ad_backend=ad, integrate=plan)
    @test counts(ad) == prepared
    @test p !== q && p.u !== q.u && p.factor !== q.factor
    @test p.plan === plan && q.plan === plan
    @test p.backend === rk && q.backend === rk
    @test p.tol == 1e-20 && p.maxiter == 70
    @test q.tol == 1e-24 && q.maxiter == 200
    @test p.factor[] === nothing && q.factor[] === nothing
    @test all(iszero, p.u) && all(iszero, q.u)
    # The derived-observation preparation uses one direction buffer tuple per
    # group. Restriction must retain that bounded representation, with buffers
    # private to each target and the original prepared AD direction arity.
    for (pg, qg, template) in zip(p.cells, q.cells, plan.prepared.problem.cells)
        hasproperty(pg, :dirs) || continue
        @test !any(c -> hasproperty(c, :dirs), pg.cells)
        @test length(pg.dirs) == length(template.dirs)
        @test all(all(iszero, d) for d in pg.dirs)
        @test all(a !== b && a !== t for (a, b, t) in zip(pg.dirs, qg.dirs, template.dirs))
    end
    if partial && cellkind === :sum
        # Reclassification must retain shared read-only maps across repeated
        # summands rather than reintroducing an array per occurrence.
        maps = [getproperty(t, field) for g in p.cells for c in g.cells
            for reduction in c.terms for t in reduction for field in (:lpos, :pidx, :ppos, :loc)]
        representative = Dict{Vector{Int},Vector{Int}}()
        @test all(get!(representative, v, v) === v for v in maps)
        @test length(representative) < length(maps)
    end
    # The old three-field constructor still works and forces an independently
    # prepared control. It also measures the duplicate setup this fix removes.
    legacy = BRMLaplacePlan(plan.terms, plan.ess_floor, plan.draws)
    baseline = @timed brm_laplace_problem(rk; ad_backend=ad, integrate=legacy)
    fresh = baseline.value
    @test counts(ad) == 2 .* prepared
    @test p.integrated == fresh.integrated
    @test p.population == fresh.population && p.names == fresh.names
    @test p.components == fresh.components
    for scale in (0.05, 0.15)
        phi = scale .* randn(Xoshiro(28), length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, phi)
        expected, expected_gradient = LogDensityProblems.logdensity_and_gradient(fresh, phi)
        @test isfinite(value) && all(isfinite, gradient)
        @test value ≈ expected atol=1e-9 rtol=1e-9
        @test gradient ≈ expected_gradient atol=2e-8 rtol=2e-8
        @test q.factor[] === nothing
        # Independent finite differences also exercise newly sampled effects.
        fd = map(eachindex(phi)) do i
            e = zeros(length(phi)); e[i] = 1e-5
            (LogDensityProblems.logdensity(q, phi + e) -
                LogDensityProblems.logdensity(q, phi - e)) / 2e-5
        end
        @test gradient ≈ fd atol=3e-6 rtol=3e-6
        q.factor[] = nothing
    end
    @test plan.prepared.problem.factor[] !== p.factor[]
    @test plan.prepared.problem.plan === nothing
    before = counts(ad)
    copied = deepcopy(p)
    @test copied.factor[] === nothing
    @test counts(ad) == before
    phi = zeros(length(p.population))
    @test LogDensityProblems.logdensity_and_gradient(copied, phi)[2] ≈
        LogDensityProblems.logdensity_and_gradient(p, phi)[2]
    println("plan reuse partial=", partial, " kind=", cellkind,
        " prepare_calls=", prepared, " reused_calls=(0, 0, 0)",
        " reused_seconds=", elapsed.time, " fresh_seconds=", baseline.time,
        " reused_bytes=", elapsed.bytes, " fresh_bytes=", baseline.bytes)
    (; plan, ad, p)
end

@stestset "row plan reuse and strict subsets" begin
    d = (; a=repeat(["a", "b", "c"], inner=4), b=repeat(["x", "y"], 6),
        y=[1, 0, 2, 1, 3, 1, 2, 0, 1, 2, 1, 0])
    builder = @brm begin
        log(mu) ~ 1 + (1 | a) + (1 | b)
        y ~ Poisson(mu)
    end
    rk = RKBRMI(builder(d))
    check_plan_reuse(rk, (:a, :b))
    (; plan, ad) = check_plan_reuse(rk, (:a, :b); partial=true)
    # A plan may still select terms in another bound backend, but must not use
    # the original backend's data. Different AD configurations also prepare.
    changed = RKBRMI(builder(merge(d, (; y=d.y .+ 2))))
    before = counts(ad)
    target = brm_laplace_problem(changed; ad_backend=ad, integrate=plan)
    @test counts(ad)[1] == before[1] + 1
    @test target.backend === changed
    other = CountingLaplaceAD()
    brm_laplace_problem(rk; ad_backend=other, integrate=plan)
    @test other.rows == 1
end

@stestset "ordinary plate plan reuse and strict subsets" begin
    d = (; a=["a", "a", "b", "b"], b=["x", "y", "x", "y"],
        y=[0.2, -0.1, 0.4, 0.3])
    rk = RKBRMI(@brm d begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | a) + (1 | b)
        @plate for k in eachindex(y)
            y[k] ~ Normal(mu[k], sigma)
        end
    end)
    for partial in (false, true)
        check_plan_reuse(rk, (:a, :b); partial, cellkind=:ordinary)
    end
end

@stestset "nested vector plate plan reuse and strict subsets" begin
    # RK may factor the vector likelihood's reduction. Both preparation paths
    # must save setup calls without changing the marginal or sharing state.
    rk = RKBRMI(nested_model(NESTED_DATA))
    for partial in (false, true)
        check_plan_reuse(rk, (:subject, :unit); partial, cellkind=:nested)
    end
end

@stestset "sum factor plan reuse and strict subsets" begin
    d = (; a=["a", "b", "c", "d"], b=["x", "x", "y", "y"],
        rows=[[1, 2], [2, 3, 4], Int[]], anchor=[1, 3, 4], y=[0.2, -0.1, 0.4])
    rk = RKBRMI(@brm d begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | a) + (1 | b)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(sum(mu[rows[k]]) + mu[anchor[k]], sigma)
        end
    end)
    for partial in (false, true)
        check_plan_reuse(rk, (:a, :b); partial, cellkind=:sum)
    end
end

@stestset "field blocks survive plan restriction together" begin
    d = (; area=repeat(["a", "b", "c", "d"], inner=2),
        group=repeat(["x", "y"], 4), from=["a", "b", "c"], to=["b", "c", "d"],
        y=[1, 2, 0, 1, 2, 1, 0, 2])
    rk = RKBRMI(@brm d begin
        log(mu) ~ 1 + car(area; edges=(from, to), type=:bym2) + (1 | group)
        y ~ Poisson(mu)
    end)
    # Retaining the field keeps both its structured and independent parts;
    # dropping it keeps their complete prior in the sampled population.
    check_plan_reuse(rk, (:area, :group); partial=true)
    check_plan_reuse(rk, (:group, :area); partial=true)
end
