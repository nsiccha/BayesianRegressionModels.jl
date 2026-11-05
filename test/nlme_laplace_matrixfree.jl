# RUN: julia --project=test test/nlme_laplace_matrixfree.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, SparseArrays, LogDensityProblems, Random, Statistics

function mf_fd(f, x; h=1e-5)
    map(eachindex(x)) do j
        d = zeros(length(x)); d[j] = h
        (f(x + d) - f(x - d)) / (2h)
    end
end

# A deliberate counter-control: vary outer coordinates while freezing the
# conditional mode. Its determinant derivative omits the implicit correction.
function mf_frozen_mode_value(p, x, z)
    pop = BRM._brm_spl_population(p,x)
    fc = zeros(length(p.components)); g=zeros(length(z))
    h = BRM._brm_spl_precision(p)
    BRM._brm_spl_assemble!(fc,g,h,p,pop.A,pop.C,z; derivatives=true,uA=pop.uA)
    F = BRM._brm_spl_factor!(p,h)
    BRM._brm_spl_value(p,(; finite=true,pop...,z,F,W=BRM._brm_spl_woodbury(p,F)))
end

function mf_crossed(; counts=false)
    a = repeat(["a$i" for i in 1:4]; inner=3)
    b = repeat(["b$i" for i in 1:3]; outer=4)
    x = collect(range(-1, 1; length=12))
    data = (; a, b, x, y=counts ? [0,1,2,1,3,1,0,2,1,3,1,2] : 0.2 .+ 0.3x)
    RKBRMI(if counts
        @brm data begin
            log(mu) ~ 1 + x + (1 | a) + (1 | b)
            y ~ Poisson(mu)
        end
    else
        @brm data begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | a) + (1 | b)
            y ~ Normal(mu, sigma)
        end
    end)
end

@stestset "matrix-free crossed conditional targets are reproducible" begin
    for counts in (false, true)
        rk = mf_crossed(; counts)
        p = brm_laplace_problem(rk; integrate=(:a,:b), ad_backend=NLME_AD,
            solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=4))
        exact = brm_laplace_problem(rk; integrate=(:a,:b), ad_backend=NLME_AD, solver=:exact)
        x = fill(0.1, length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
        @test isfinite(value)
        @test gradient ≈ mf_fd(y -> LogDensityProblems.logdensity(p,y), x) rtol=2e-5 atol=2e-5
        @test value == LogDensityProblems.logdensity(p, x)
        LogDensityProblems.logdensity(p, x .+ 0.15)
        @test LogDensityProblems.logdensity_and_gradient(p,x) == (value,gradient)
        copy_p = deepcopy(p)
        @test copy_p.algebra.probes == p.algebra.probes
        @test copy_p.algebra.probes !== p.algebra.probes
        @test copy_p.algebra.references !== p.algebra.references
        @test all(zip(copy_p.algebra.references,p.algebra.references)) do (a,b)
            a.lower == b.lower && a.lower !== b.lower
        end
        @test LogDensityProblems.logdensity_and_gradient(copy_p,x) == (value,gradient)
        if counts
            z = copy(BRM._brm_spl_mode!(p,x).z)
            frozen_gradient = mf_fd(y -> mf_frozen_mode_value(p,y,z),x)
            @test norm(gradient-frozen_gradient) > 1e-4
        else
            P = repeat(x,1,4)
            draws = brm_laplace_draws(p,P; rng=Xoshiro(76)).draws
            @test size(draws) == (length(p.u),4) && all(isfinite,draws)
            @test draws == brm_laplace_draws(copy_p,P; rng=Xoshiro(76)).draws
        end
        @test exact.algebra === nothing
        @test p.factor[] === nothing
        # Hadamard probes remove Monte Carlo error for this seven-coordinate
        # control, leaving only the finite rational and iterative errors.
        hadamard = [1.0;;]
        for _ in 1:3
            hadamard = [hadamard hadamard; hadamard -hadamard]
        end
        BRM._brm_spl_algebra!(p, :matrixfree, MatrixFreeLaplace(; probes=8))
        p.algebra.probes[1] .= hadamard[1:7, :]
        v, g = LogDensityProblems.logdensity_and_gradient(p, x)
        ve, ge = LogDensityProblems.logdensity_and_gradient(exact, x)
        @test v ≈ ve atol=3e-7
        @test g ≈ ge rtol=3e-6 atol=3e-6
        default = brm_laplace_problem(rk; integrate=(:a,:b), ad_backend=NLME_AD)
        @test default.algebra === nothing
        auto = brm_laplace_problem(rk; integrate=(:a,:b), ad_backend=NLME_AD,
            matrixfree=MatrixFreeLaplace(; exact_limit=4))
        @test auto.algebra.large == [true]
    end
end

@stestset "matrix-free numerical policy survives plans and ladder replanning" begin
    rk = mf_crossed()
    options = MatrixFreeLaplace(; exact_limit=4, probes=2, seed=91)
    plan = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=(:a,:b),
        draws=8, solver=:matrixfree, matrixfree=options)
    p = brm_laplace_problem(rk; integrate=plan, ad_backend=NLME_AD)
    @test p.plan === plan
    @test p.algebra.options === options
    @test p.algebra.solver === :matrixfree
    x = fill(0.1, length(p.population))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p,x)
    q, next_plan, switched = BRM._brm_ladder_replan(p, copy(p.u);
        ad_backend=NLME_AD, rng=Xoshiro(43))
    @test isempty(switched)
    @test next_plan.algebra_policy === plan.algebra_policy
    @test q.algebra.probes == p.algebra.probes
    @test LogDensityProblems.logdensity_and_gradient(q,x) == (value,gradient)
    exact = brm_laplace_problem(rk; integrate=plan, ad_backend=NLME_AD, solver=:exact)
    @test exact.algebra === nothing
    @test exact.plan.algebra_policy.solver === :exact
    @test plan.algebra_policy.solver === :matrixfree
    fit = brm_ladder_fit(rk; ad_backend=NLME_AD, integrate=plan, upto=1,
        start=x, draws=1, rng=Xoshiro(44))
    @test fit.problem.algebra.options === options
    continued = brm_ladder_fit(rk; ad_backend=NLME_AD, continue_from=fit, upto=1,
        draws=1, rng=Xoshiro(45))
    @test continued.problem === fit.problem
    @test continued.plan === fit.plan
end

@stestset "matrix-free nonlinear ordinary cells and rank-deficient sensitivities" begin
    rk = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    p = brm_laplace_problem(rk; integrate=:subject, ad_backend=NLME_AD,
        solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=1))
    x = zeros(length(p.population))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
    @test isfinite(value)
    @test gradient ≈ mf_fd(y -> LogDensityProblems.logdensity(p,y), x; h=1e-4) rtol=2e-4 atol=2e-4
    @test p.factor[] === nothing
    S = ones(4,4)
    L = BRM._brm_spl_psd_root(S)
    @test L * L' ≈ S atol=1e-12
end

@stestset "matrix-free nonlinear overlapping sums retain mixed derivatives" begin
    data = (; unit=["u$i" for i in 1:7], rows=[[1,2,2],[2,3,4],[4,5,6,7]],
        anchor=[1,2,4], y=[3.7,3.7,4.9])
    rk = RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        eta ~ 1 + (1 | unit)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(sum(exp.(eta[rows[k]])) +
                0.2sum(eta[rows[k]].^2) + eta[anchor[k]], sigma)
        end
    end)
    p = brm_laplace_problem(rk; integrate=:unit, ad_backend=NLME_AD,
        solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=4))
    for shift in (0.0, 0.2)
        x = fill(shift, length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
        @test isfinite(value)
        @test gradient ≈ mf_fd(y -> LogDensityProblems.logdensity(p,y), x; h=1e-4) rtol=2e-4 atol=2e-4
        state = BRM._brm_spl_mode!(p, x)
        @test state.h isa BRM._BRMSumPrecision
        @test all(part -> part.factor === nothing, state.F.parts)
        @test p.factor[] === nothing
    end
end

@stestset "matrix-free mode search handles moving negative-binomial shape" begin
    data = (; a=repeat(["a$i" for i in 1:4]; inner=3),
        b=repeat(["b$i" for i in 1:3]; outer=4),
        x=collect(range(-1,1; length=12)), y=[0,1,2,1,3,1,0,2,1,3,1,2])
    rk = RKBRMI(@brm data begin
        log(mu) ~ 1 + x + (1 | mean | a) + (1 | b)
        log(shape) ~ 1 + (1 | dispersion | a)
        y ~ NegativeBinomial2(mu,shape)
    end)
    p = brm_laplace_problem(rk; integrate=(:a,:b), ad_backend=NLME_AD,
        solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=4))
    scales = findall(m -> occursin(".tau.",String(m)),p.names)
    for s in (1.0,2.0,3.0)
        x = zeros(length(p.population)); x[scales] .= s
        value, gradient = LogDensityProblems.logdensity_and_gradient(p,x)
        @test isfinite(value) && all(isfinite,gradient)
        @test gradient ≈ mf_fd(y -> LogDensityProblems.logdensity(p,y),x; h=1e-4) rtol=2e-4 atol=2e-4
    end
    @test p.nonfinite == 0
    # The search record of the last evaluation, one entry per component.
    @test length(p.search.stages) == BRM._BRM_LAP_STAGES
    @test all(in((:tol, :floor)), p.search.stop)
    @test size(p.search.stage_shifts) == (length(p.components), BRM._BRM_LAP_STAGES)
    @test all(p.search.steps .>= p.search.backtracks .>= 0)
end

@stestset "matrix-free declared Gaussian priors retain moving mean and scale derivatives" begin
    data=(; left=[1,2,3,1],right=[2,3,1,3],y=[0.9,1.2,0.7,1.1])
    rk=RKBRMI(@brm data begin
        offset ~ Normal(0,1)
        location ~ Normal(0.2,0.8)
        scale ~ LogNormal(0,0.4)
        field[1:3] .~ Normal.(location,scale)
        @plate for i in eachindex(y)
            y[i] ~ Normal(offset + exp(field[left[i]] + 0.3field[right[i]]),0.8)
        end
    end)
    p=brm_laplace_problem(rk; integrate=(:field,),ad_backend=NLME_AD,
        solver=:matrixfree,matrixfree=MatrixFreeLaplace(;probes=2))
    for point in (-0.1,0.15)
        x=fill(point,length(p.population))
        value,gradient=LogDensityProblems.logdensity_and_gradient(p,x)
        @test isfinite(value)
        @test gradient ≈ mf_fd(y -> LogDensityProblems.logdensity(p,y),x) rtol=2e-5 atol=2e-5
    end
    x=fill(0.1,length(p.population))
    reference=LogDensityProblems.logdensity_and_gradient(p,x)
    LogDensityProblems.logdensity(p,x .+ 0.3)
    @test LogDensityProblems.logdensity_and_gradient(p,x) == reference
    BRM._brm_spl_algebra!(p,:matrixfree,MatrixFreeLaplace(;probes=4))
    p.algebra.probes[1] .= [1.0 1 1 1; 1 -1 1 -1; 1 1 -1 -1]
    exact=brm_laplace_problem(rk; integrate=(:field,),ad_backend=NLME_AD,solver=:exact)
    value,gradient=LogDensityProblems.logdensity_and_gradient(p,x)
    ve,ge=LogDensityProblems.logdensity_and_gradient(exact,x)
    @test value ≈ ve atol=5e-7
    @test gradient ≈ ge rtol=3e-6 atol=3e-6
end
