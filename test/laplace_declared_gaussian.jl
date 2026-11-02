# RUN: julia --project=test test/laplace_declared_gaussian.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, SparseArrays, LogDensityProblems, Random

function declared_fd(f, x; h=1e-5)
    [begin
        e = zeros(length(x)); e[j] = h
        (f(x+e)-f(x-e))/(2h)
    end for j in eachindex(x)]
end

function declared_fixture(; moving=false, nonlinear=false, standard=false, n=4)
    data = (; y=[0.2, -0.4, 0.8, 0.3], idx=[1,2,1,4], n,
        prior_mean=standard ? zeros(n) : collect(range(-0.3, 0.7; length=n)),
        prior_scale=standard ? ones(n) : collect(range(0.5, 1.4; length=n)))
    model = if moving
        @brm data begin
            offset ~ Normal(0, 1)
            location ~ Normal(0.2, 0.8)
            scale ~ LogNormal(0, 0.4)
            field[1:n] .~ Normal.(location, scale)
            prediction = offset + field[idx]
            y ~ Normal(prediction, 0.6)
        end
    elseif nonlinear
        @brm data begin
            offset ~ Normal(0, 1)
            location ~ Normal(0.2, 0.8)
            scale ~ LogNormal(0, 0.4)
            field[1:n] .~ Normal.(location, scale)
            prediction = offset + exp(field[idx])
            y ~ Normal(prediction, 0.6)
        end
    else
        @brm data begin
            offset ~ Normal(0, 1)
            field[1:n] .~ Normal.(prior_mean, prior_scale)
            prediction = offset + field[idx]
            y ~ Normal(prediction, 0.6)
        end
    end
    RKBRMI(model), data
end

@stestset "declared Gaussian exact marginal and outer gradients" begin
    for (moving, standard) in ((false,true), (false,false), (true,false))
        rk, data = declared_fixture(; moving, standard)
        plan = brm_laplace_plan(rk; ad_backend=NLME_AD, draws=12)
        @test only(plan.terms).admitted
        @test only(plan.terms).declaration === :field
        @test only(plan.terms).importance_ess.min ≈ 12 atol=1e-8
        @test plan.integrated_dimension == 4
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
        q = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:field,))
        @test p.population == q.population
        @test nnz(p.pattern) == 4
        read = BRM._rk_value_reader(rk, moving ? (:offset,:location,:scale) : (:offset,))
        B = [Float64(j == data.idx[i]) for i in eachindex(data.y), j in 1:data.n]
        function exact(phi)
            u = zeros(length(p.u)); u[p.population] .= phi
            values = read(u)
            means = moving ? fill(values.location, data.n) : data.prior_mean
            scales = moving ? fill(values.scale, data.n) : data.prior_scale
            covariance = 0.6^2 * I + B * Diagonal(scales.^2) * B'
            result = logpdf(MvNormal(values.offset .+ B*means, Matrix(covariance)), data.y)
            result += logpdf(Normal(0,1), values.offset)
            if moving
                result += logpdf(Normal(0.2,0.8), values.location)
                result += logpdf(LogNormal(0,0.4), values.scale) + log(values.scale)
            end
            result
        end
        for amplitude in (-0.3, 0.0, 0.4)
            phi = amplitude .* collect(1:length(p.population))
            value, gradient = LogDensityProblems.logdensity_and_gradient(p, phi)
            @test value ≈ exact(phi) atol=1e-8
            @test gradient ≈ declared_fd(exact, phi) atol=2e-6 rtol=2e-6
            @test LogDensityProblems.logdensity(q, phi) ≈ value atol=1e-9
        end
        sampled = brm_laplace_plan(rk; ad_backend=NLME_AD, sample=:field)
        @test sampled.integrated_dimension == 0
        @test only(sampled.terms).class === :sampled
        @test_throws "both forced and sampled" brm_laplace_plan(rk;
            ad_backend=NLME_AD, integrate=:field, sample=:field)
        # Copies own prior caches, selected factors and cell buffers.
        copy_p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
        @test p.gaussian.mean !== copy_p.gaussian.mean
        @test p.gaussian.values !== copy_p.gaussian.values
        @test p.factor !== copy_p.factor
        a = fill(0.1, length(p.population)); b = fill(-0.2, length(a))
        va = LogDensityProblems.logdensity_and_gradient(p, a)
        LogDensityProblems.logdensity_and_gradient(copy_p, b)
        @test LogDensityProblems.logdensity_and_gradient(p, a) == va
    end
end

@stestset "generated effects coexist with declared vectors and retained subsets" begin
    data = (; g=["a","b","a","c"], idx=[1,2,1,3], y=[0.2,-0.1,0.4,0.7])
    rk = RKBRMI(@brm data begin
        eta ~ 1 + (1 | g)
        location ~ Normal(0.2, 0.8)
        scale ~ LogNormal(0, 0.4)
        field[1:3] .~ Normal.(location, scale)
        prediction = eta + field[idx]
        y ~ Normal(prediction, 0.7)
    end)
    for keep in (:g,:field)
        plan = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=keep,
            ess_floor=1.1, draws=8)
        @test count(t -> t.admitted, plan.terms) == 1
        @test plan.integrated_dimension == 3
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
        q = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(keep,))
        phi = 0.07 .* collect(1:length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, phi)
        reference, refgradient = LogDensityProblems.logdensity_and_gradient(q, phi)
        @test value ≈ reference atol=1e-9
        @test gradient ≈ refgradient atol=1e-8
        @test gradient ≈ declared_fd(x -> LogDensityProblems.logdensity(p,x), phi) atol=2e-6 rtol=2e-6
    end
end

@stestset "prior dependence stays outer and non-Gaussian vectors stay sampled" begin
    data = (; y=[0.2,0.4], idx=[1,2])
    rk = RKBRMI(@brm data begin
        parent[1:2] .~ Normal.(0,1)
        child[1:2] .~ Normal.(parent,0.7)
        positive[1:2] .~ LogNormal.(0,1)
        prediction = child[idx] + positive[idx]
        y ~ Normal(prediction,0.6)
    end)
    plan = brm_laplace_plan(rk; ad_backend=NLME_AD, draws=8)
    @test !only(t for t in plan.terms if t.declaration === :parent).admitted
    @test only(t for t in plan.terms if t.declaration === :child).admitted
    @test only(t for t in plan.terms if t.declaration === :positive).class === :never
    @test_throws "must stay sampled" brm_laplace_plan(rk;
        ad_backend=nothing, integrate=:parent)
    @test_throws "independent Normal" brm_laplace_plan(rk;
        ad_backend=nothing, integrate=:positive)
end

@stestset "nonlinear Normal observations are diagnosed and differentiated" begin
    rk, _ = declared_fixture(; nonlinear=true)
    plan = brm_laplace_plan(rk; ad_backend=NLME_AD, draws=20, ess_floor=0.001)
    @test only(plan.terms).class === :gated
    @test only(plan.terms).importance_ess !== nothing
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:field,))
    for point in (-0.15, 0.2)
        phi = fill(point, length(p.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, phi)
        @test isfinite(value)
        @test gradient ≈ declared_fd(x -> LogDensityProblems.logdensity(p,x), phi) atol=2e-5 rtol=2e-5
    end
end

@stestset "declared Gaussian footprint width and linear sparse storage" begin
    data = (; n=8, y=[0.3], rows=[collect(1:8)], weights=ones(8))
    rk = RKBRMI(@brm data begin
        field[1:n] .~ Normal.(0, 1)
        offset ~ Normal(0, 1)
        @plate for i in eachindex(rows)
            y[i] ~ Normal(offset + dot(field[rows[i]], weights), 0.7)
        end
    end)
    plan = brm_laplace_plan(rk; ad_backend=nothing, max_cell_width=4)
    @test !only(plan.terms).admitted
    @test plan.integrated_dimension == 0
    @test plan.width_events[:field].contribution == 8
    @test_throws "max_cell_width=4" brm_laplace_plan(rk;
        ad_backend=nothing, integrate=:field, max_cell_width=4)
    @test_throws "max_cell_width=4" brm_laplace_problem(rk;
        ad_backend=nothing, integrate=(:field,), max_cell_width=4)
    for n in (16,64)
        sparse_rk, _ = declared_fixture(; n)
        p = brm_laplace_problem(sparse_rk; ad_backend=NLME_AD, integrate=(:field,))
        @test length(p.integrated) == n
        @test nnz(p.pattern) == n
        @test length(p.gaussian.mean) == n
        @test all(c -> length(c.lz) == 1, only(p.cells).cells)
    end
end
