include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, SparseArrays, LogDensityProblems, Random, Statistics, Optim

function sum_cell_fixture(n; rows=[collect(1:n)], nonlinear=false, mixed=false)
    data = (; unit=["u$i" for i in 1:n], rows,
        anchor=[isempty(r) ? 1 : first(r) for r in rows],
        y=[(nonlinear ? 1.2length(r) : 0.2length(r)) + 0.1 for r in rows])
    brmi = if mixed
        @brm data begin
            sigma ~ Exponential(1)
            eta ~ 1 + (1 | unit)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(sum(exp.(eta[rows[k]])) +
                    0.2sum(eta[rows[k]].^2) + eta[anchor[k]], sigma)
            end
        end
    elseif nonlinear
        @brm data begin
            sigma ~ Exponential(1)
            eta ~ 1 + (1 | unit)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(sum(exp.(eta[rows[k]])), sigma)
            end
        end
    else
        @brm data begin
            sigma ~ Exponential(1)
            eta ~ 1 + (1 | unit)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(sum(eta[rows[k]]), sigma)
            end
        end
    end
    (; rk=RKBRMI(brmi), data)
end

@stestset "indexed direct sum-cell footprints stay compact" begin
    for n in (16, 64, 4096)
        (; rk) = sum_cell_fixture(n; mixed=true)
        factors = prepare_cell_factors(rk.model, rk_bound_plan(rk), :y; locality=:error)
        outer = prepare_cell_factor_query(factors, :outer)
        summand = prepare_cell_factor_query(factors, 1)
        outer_footprint = partition_footprint_kernel(outer.kernel)
        summand_footprint = partition_footprint_kernel(summand.kernel)
        u = zeros(rk.model.layout.total)
        sums = zeros(length(cell_reductions(factors)))
        direct = Base.invokelatest(outer_footprint, u, sums, 1).unconstrained
        term = Base.invokelatest(summand_footprint, u, 1, 1).unconstrained
        @test length(direct) == 4
        @test length(term) == 3
    end
end

function sum_fd(f, x; h=1e-5)
    map(eachindex(x)) do i
        e = zeros(length(x)); e[i] = h
        (f(x + e) - f(x - e)) / (2h)
    end
end

@stestset "sum cells integrate exact Gaussian marginals and conditional draws" begin
    rows = [[1,2,2], [2,3,4], Int[], [5,6]]
    (; rk, data) = sum_cell_fixture(6; rows)
    p = brm_laplace_problem(rk; integrate=(:unit,), ad_backend=NLME_AD)
    @test only(p.cells) isa BRM._BRMSumCellGroup
    @test nnz(p.pattern) == 6
    @test length(p.components) == 2
    φ = 0.05randn(Xoshiro(15), length(p.population))
    function exact(φ)
        u = zeros(length(p.u)); u[p.population] .= φ
        values = BRM._rk_constrained_values(rk, u)
        plan = rk_bound_plan(rk)
        prior = prepare_query(rk.model, plan, :prior)
        jac = prepare_query(rk.model, plan, :log_jacobian)
        popprior = Base.invokelatest(prior,u) + Base.invokelatest(jac,u) -
            length(p.integrated) * logpdf(Normal(), 0.0)
        τ = only(values.b_unit.tau)
        μ = length.(rows) .* values.eta_Intercept
        A = [count(==(i), row) for row in rows, i in 1:6]
        V = τ^2 * A * A' + values.sigma^2 * I
        popprior + logpdf(MvNormal(μ, Symmetric(V)), data.y)
    end
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
    @test value ≈ exact(φ) atol=1e-9
    @test gradient ≈ sum_fd(exact, φ) rtol=2e-6 atol=2e-6
    state = BRM._brm_spl_mode!(p,φ)
    H = Matrix(state.h.D + state.h.B' * state.h.W * state.h.B)
    @test state.F.logdet ≈ logdet(Symmetric(H)) atol=1e-9
    @test sum(BRM._brm_spl_component_logdets(p,state.F)) ≈ state.F.logdet atol=1e-9
    sets = [[i] for i in 1:6]
    @test only.(BRM._brm_spl_inverse_blocks(p,state,sets)) ≈ diag(inv(H))
    draws = reduce(hcat, (BRM._brm_spl_draw(p,state,Xoshiro(i)) for i in 1:2500))
    @test maximum(abs, vec(mean(draws;dims=2))) < 0.07
    @test cov(draws;dims=2) ≈ inv(H) atol=0.07
    out = brm_laplace_draws(p, repeat(φ,1,3); conditional=12, rng=Xoshiro(52))
    @test out.diagnostics.importance_ess ≈ fill(12.0,2,3) atol=1e-8
    cp = deepcopy(p)
    @test cp.factor[] === nothing
    @test LogDensityProblems.logdensity_and_gradient(cp,φ)[2] ≈ gradient
    @test cp.factor[].F !== p.factor[].F
    plan = brm_laplace_plan(rk; ad_backend=NLME_AD, draws=8, at=φ)
    @test only(plan.terms).admitted
    @test only(plan.terms).importance_ess.min ≈ 8.0 atol=1e-7
    fit = brm_laplace_mode(p; init=φ, iterations=200)
    @test fit.diagnostics.converged
    @test fit.diagnostics.value ≈ exact(fit.draws[p.population,1]) atol=1e-8
    @test norm(sum_fd(exact,fit.draws[p.population,1]),Inf) < 1e-4
end

@stestset "nonlinear overlapping sum-cell modes and marginal gradients" begin
    (; rk) = sum_cell_fixture(7; rows=[[1,2,2],[2,3,4],[4,5,6,7]], nonlinear=true, mixed=true)
    p = brm_laplace_problem(rk; integrate=(:unit,), ad_backend=NLME_AD)
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    for φ in (zeros(length(p.population)), 0.1randn(Xoshiro(21),length(p.population)))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p,φ)
        @test isfinite(value)
        @test gradient ≈ sum_fd(x -> LogDensityProblems.logdensity(p,x),φ) rtol=1e-4 atol=1e-5
        state = BRM._brm_spl_mode!(p,φ)
        joint, g = LogDensityProblems.logdensity_and_gradient(full,state.u)
        @test norm(g[p.integrated],Inf) < 1e-6
        H = -reduce(hcat, map(p.integrated) do j
            e = zeros(length(state.u)); e[j] = 1e-5
            gp = LogDensityProblems.logdensity_and_gradient(full,state.u+e)[2]
            gm = LogDensityProblems.logdensity_and_gradient(full,state.u-e)[2]
            (gp[p.integrated] - gm[p.integrated]) / 2e-5
        end)
        @test Matrix(state.h.D + state.h.B' * state.h.W * state.h.B) ≈ H rtol=1e-7 atol=1e-7
        @test value ≈ joint + length(p.integrated)/2*log(2π) - logdet(Symmetric(H))/2 atol=1e-7
    end
    out = brm_laplace_draws(p, zeros(length(p.population),2); conditional=8, rng=Xoshiro(53))
    @test all(isfinite,out.draws)
    @test all(1 .<= out.diagnostics.importance_ess .<= 8.0000001)
end

@stestset "sum cells retain ICAR soft constraints and local diagnostic densities" begin
    area = ["a$i" for i in 1:6]
    data = (; area, from=area[1:5], to=area[2:6],
        rows=[[1,2,3], [3,4,5,6]], y=[0.4, -0.2])
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        eta ~ 1 + car(area; edges=(from,to), type=:icar)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(sum(eta[rows[k]]), sigma)
        end
    end)
    p = brm_laplace_problem(rk; integrate=(:area,), ad_backend=NLME_AD)
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    phi = 0.05randn(Xoshiro(71), length(p.population))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p,phi)
    state = BRM._brm_spl_mode!(p,phi)
    @test length(p.wcols) == 1
    @test state.h isa BRM._BRMSumPrecision
    @test rank(Matrix(state.h.D)) == length(p.integrated)-1
    @test gradient ≈ sum_fd(x -> LogDensityProblems.logdensity(p,x),phi) rtol=1e-5 atol=1e-6
    joint, g = LogDensityProblems.logdensity_and_gradient(full,state.u)
    @test norm(g[p.integrated],Inf) < 1e-6
    H = -reduce(hcat, map(p.integrated) do j
        e = zeros(length(state.u)); e[j] = 1e-5
        gp = LogDensityProblems.logdensity_and_gradient(full,state.u+e)[2]
        gm = LogDensityProblems.logdensity_and_gradient(full,state.u-e)[2]
        (gp[p.integrated] - gm[p.integrated]) / 2e-5
    end)
    @test Matrix(state.h.D + state.h.B' * state.h.W * state.h.B) ≈ H rtol=1e-7 atol=1e-7
    @test value ≈ joint + length(p.integrated)/2*log(2π) - logdet(Symmetric(H))/2 atol=1e-7
    # A block's cached local density must change by the whole model's density
    # change, including every touching graph edge and the global soft sum.
    cache = BRM._brm_plan_local_cache(p,state)
    for i in eachindex(state.z)
        terms = [c.term for group in p.cells for c in group.cells if i in c.lz]
        z = copy(state.z); u = copy(state.u)
        before = BRM._brm_plan_local_value(p,state,terms,[i],z,u,cache)
        z[i] += 0.13
        after = BRM._brm_plan_local_value(p,state,terms,[i],z,u,cache)
        u[p.integrated] .= z
        @test after-before ≈ LogDensityProblems.logdensity(full,u)-joint atol=1e-8
    end
    ess = BRM._brm_plan_ess(p,state,only(p.blocks).declaration,8,0.5,Xoshiro(72))
    @test ess.min ≈ 8.0 atol=1e-7
    draws = brm_laplace_draws(p,reshape(phi,:,1); conditional=8,rng=Xoshiro(73))
    @test only(draws.diagnostics.importance_ess) ≈ 8.0 atol=1e-7
end

@stestset "pivoted augmented factors certify every scalar curvature sign" begin
    previous = nothing
    for d in (-1.0,0.0,1.0,2.0), b in (0.1,1.0,2.0), w in (-2.0,-1.0,0.0,1.0,2.0,4.0)
        H = d + b*b*w
        K = BRM._brm_spl_augmented_matrix(sparse([d;;]),sparse([b;;]),sparse([w;;]))
        f = BRM._brm_spl_augmented_factor(K,1; previous)
        @test f.positive_definite == (H > 0)
        if H > 0
            @test (f \ [1.0])[1] ≈ 1/H
            @test f.logdet ≈ log(H) atol=1e-10
            @test BRM._brm_spl_selected(f)[1,1] ≈ 1/H
        end
        previous = f
    end
end
