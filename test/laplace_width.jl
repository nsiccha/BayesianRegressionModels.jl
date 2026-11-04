include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, SparseArrays, LogDensityProblems, Random, Optim, Pathfinder

@stestset "sparse pattern deduplicates before enumerating pairs" begin
    for width in (8, 32, 128), cells in (1, 40, 400)
        footprints = [collect(1:width) for _ in 1:cells]
        out = BRM._brm_spl_pattern(footprints, width + 2, [(width + 1, width + 2)])
        @test out.stats.input_indices == cells * width
        @test out.stats.distinct_footprints == 1
        @test out.stats.pair_visits == width^2
        @test out.stats.retained_nnz == width^2 + 4
        @test out.pattern.colptr == sparse(ones(width + 2, width + 2) .* [
            (i <= width && j <= width) || (i > width && j > width)
            for i in 1:(width + 2), j in 1:(width + 2)]).colptr
    end
    out = BRM._brm_spl_pattern([[1,2], [2,3], [1,2], Int[]], 4, Tuple{Int,Int}[])
    @test out.stats.pair_visits == 8
    @test out.stats.retained_nnz == 8
end

@stestset "whole-term width policy is stable and updates overlapping cells" begin
    c(cell, counts) = (; observation=:y, cell, part=:cell, extra=0, counts)
    constraints = [c(1, [1=>4, 2=>4]), c(2, [1=>4, 3=>2]), c(3, [2=>4, 3=>2])]
    s = BRM._brm_spl_select_width(constraints, [:a,:b,:c], Set{Symbol}(), 6)
    @test s.active == [false,true,true]
    @test s.events[:a] == (; observation=:y, cell=1, part=:cell, width=8, bound=6, contribution=4)
    @test s.max_width == 6
    forced = BRM._brm_spl_select_width(constraints, [:a,:b,:c], Set([:a]), 6)
    @test forced.active == [true,false,true]
    @test_throws "forced terms" BRM._brm_spl_select_width(constraints,
        [:a,:b,:c], Set([:a,:b]), 6)
    multiple = BRM._brm_spl_select_width([c(1, [i => 8-i for i in 1:6])],
        [:a,:b,:c,:d,:e,:f], Set{Symbol}(), 8)
    @test multiple.active == [false,false,false,false,true,true]
    @test [multiple.events[t].width for t in (:a,:b,:c,:d)] == [27,20,14,9]
    for value in (0, -1, Inf, NaN, 1.5, true)
        @test_throws ArgumentError BRM._brm_spl_width_limit(value)
    end
end

function width_fixture(n=8; correlated=false)
    data = (; a=["a$i" for i in 1:n], b=["b$(mod1(i,2))" for i in 1:n],
        x=collect(range(-0.5,0.7; length=n)), rows=[collect(1:n), collect(1:n)],
        weights=ones(n), y=[0.2,0.3])
    model = if correlated
        @brm data begin
            eta ~ 1 + (1 + x | a) + (1 | b)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(dot(eta[rows[k]], weights), 0.7)
            end
        end
    else
        @brm data begin
            eta ~ 1 + (1 | a) + (1 | b)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(dot(eta[rows[k]], weights), 0.7)
            end
        end
    end
    RKBRMI(model)
end

@stestset "width screening retains the full model and correlated whole terms" begin
    for correlated in (false, true)
        rk = width_fixture(; correlated)
        plan = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=(:b,), max_cell_width=4, draws=8)
        a = only(t for t in plan.terms if t.grouping === :a)
        b = only(t for t in plan.terms if t.grouping === :b)
        @test !a.admitted && a.class === :sampled && b.admitted
        @test plan.integrated_dimension == 2
        @test plan.width_events[a.declaration].width == (correlated ? 18 : 10)
        @test plan.width_events[a.declaration].cell == 1
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
        q = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:b,), max_cell_width=4)
        @test length(p.integrated) == 2
        @test !hasproperty(p, :tnz)
        @test p.pattern_stats.pair_visits == 4
        @test all(g -> all(c -> length(c.lz) <= 4, g.cells), p.cells)
        phi = 0.07randn(Xoshiro(34), length(p.population))
        vp, gp = LogDensityProblems.logdensity_and_gradient(p, phi)
        vq, gq = LogDensityProblems.logdensity_and_gradient(q, phi)
        @test vp ≈ vq atol=1e-9
        @test gp ≈ gq atol=1e-9
        # Independent exact Gaussian integration of the SAME retained set,
        # using only full-model density probes at matched packed coordinates.
        full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        function exact(phi)
            u = zeros(length(p.u)); u[p.population] .= phi
            f(z) = begin
                u[p.integrated] .= z
                LogDensityProblems.logdensity(full, u)
            end
            f0 = f([0.,0.])
            plus = [f([1.,0.]), f([0.,1.])]
            minus = [f([-1.,0.]), f([0.,-1.])]
            g = (plus-minus)/2
            h = 2 * f0 .- plus .- minus
            cross = -(f([1.,1.]) - sum(plus) + f0)
            H = Symmetric([h[1] cross; cross h[2]])
            f0 + (dot(g,H\g) + 2log(2pi) - logdet(H))/2
        end
        @test vp ≈ exact(phi) atol=1e-8
        fd = map(eachindex(phi)) do j
            e = zeros(length(phi)); e[j] = 1e-5
            (exact(phi + e) - exact(phi - e)) / 2e-5
        end
        @test gp ≈ fd atol=1e-6 rtol=1e-6
        @test_throws "max_cell_width=4" brm_laplace_plan(rk;
            ad_backend=nothing, integrate=(:a,), max_cell_width=4)
        @test_throws "max_cell_width=4" brm_laplace_problem(rk;
            ad_backend=nothing, integrate=(:a,), max_cell_width=4)
    end
end

@stestset "no admitted terms supplies full density gradient draws and ladder" begin
    wide = width_fixture(129)
    default = brm_laplace_plan(wide; ad_backend=nothing, sample=(:b,))
    @test default.max_cell_width == 128 && default.integrated_dimension == 0
    @test only(values(default.width_events)).width == 129
    rk = width_fixture(4)
    # No conditional derivative preparation is possible with this sentinel.
    plan = brm_laplace_plan(rk; ad_backend=nothing, max_cell_width=1)
    @test all(t -> !t.admitted, plan.terms)
    @test plan.integrated_dimension == 0
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u = 0.1randn(Xoshiro(19), LogDensityProblems.dimension(full))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, u)
    vf, gf = LogDensityProblems.logdensity_and_gradient(full, u)
    @test value ≈ vf
    @test gradient ≈ gf
    @test plan.sampled_dimension == length(u)
    draws = repeat(u, 1, 3)
    out = brm_laplace_draws(p, draws; conditional=8)
    @test out.draws == draws
    @test out.diagnostics.newton_steps == [0,0,0]
    @test size(out.diagnostics.importance_ess) == (0,3)
    fit = brm_ladder_fit(rk; ad_backend=NLME_AD, integrate=:auto, max_cell_width=1, start=u,
        upto=4, draws=3, conditional=8)
    @test size(fit.draws) == (length(u),3)
    @test fit.ladder[1].components == 0
    @test all(isfinite, fit.draws)
    starts = brm_start(rk; ad_backend=NLME_AD, integrate=plan, chains=2,
        rng=Xoshiro(12))
    @test length(starts) == 2
    @test all(s -> length(s) == length(u) && all(isfinite, s), starts)

    # An enlarged sampled target must not allocate a dense population Hessian.
    sampler_calls = Ref(0)
    sampler(q, init, rng) = (sampler_calls[] += 1; repeat(init, 1, 2))
    large = brm_ladder_fit(wide; ad_backend=NLME_AD, integrate=default,
        upto=5, draws=2, sampler)
    curvature = only(c for c in large.ladder if c.name === :population_curvature)
    @test curvature.skipped && occursin("128", curvature.reason)
    @test sampler_calls[] == 1
    @test size(large.draws) == (default.sampled_dimension, 2)
end

@stestset "sum factors bound local directions and keep sampled gradients" begin
    data = (; a=["a$i" for i in 1:6], b=repeat(["b1","b2"],3),
        rows=[[1,2,3,4],[3,4,5,6]], y=[0.2,-0.1])
    rk = RKBRMI(@brm data begin
        eta ~ 1 + (1 | a) + (1 | b)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(sum(eta[rows[k]]), 0.7)
        end
    end)
    plan = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=(:b,), max_cell_width=1, draws=8)
    @test only(t for t in plan.terms if t.grouping === :b).admitted
    @test !only(t for t in plan.terms if t.grouping === :a).admitted
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
    group = only(p.cells)
    @test group isa BRM._BRMSumCellGroup
    @test length(p.integrated) == 2 # a sum's union may exceed its local bound
    @test all(length(t.lz) <= 1 for c in group.cells for reduction in c.terms for t in reduction)
    phi = 0.08randn(Xoshiro(89),length(p.population))
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    function exact(phi)
        u = zeros(length(p.u)); u[p.population] .= phi
        f(z) = begin
            u[p.integrated] .= z
            LogDensityProblems.logdensity(full,u)
        end
        f0 = f([0.,0.])
        plus = [f([1.,0.]), f([0.,1.])]
        minus = [f([-1.,0.]), f([0.,-1.])]
        g = (plus-minus)/2
        h = 2 * f0 .- plus .- minus
        cross = -(f([1.,1.]) - sum(plus) + f0)
        H = Symmetric([h[1] cross; cross h[2]])
        f0 + (dot(g,H\g) + 2log(2pi) - logdet(H))/2
    end
    value, gradient = LogDensityProblems.logdensity_and_gradient(p,phi)
    @test value ≈ exact(phi) atol=1e-8
    fd = map(eachindex(phi)) do j
        e = zeros(length(phi)); e[j] = 1e-5
        (exact(phi+e)-exact(phi-e))/2e-5
    end
    @test gradient ≈ fd atol=1e-6 rtol=1e-6
    state = BRM._brm_spl_mode!(p,phi)
    blocks = BRM._brm_spl_factor_blocks(group.prepared,state.u,first(group.cells))
    @test size(blocks.hf,2) == length(blocks.active)
    @test all(size(t.h,2) == length(t.lz) for reduction in blocks.terms for t in reduction)
end
