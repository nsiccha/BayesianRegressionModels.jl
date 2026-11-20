# test/rw_second_order.jl — `rw(time; order, sum_to_zero)`: the zero-start
# second-order walk and the intrinsic (sum-to-zero) walks of order 1 and 2 on
# SBBRMI and RKBRMI, their forecasts, and the sparse Laplace integrating them.
# Snag brm-no-second-or-fd336281, design decision 1ot6lwb.
#
# Covered: keyword admission and refusals; the emitted Stan programs, with
# `rw(time)` unchanged; the intrinsic law of a sum-to-zero walk's values,
# `-(n - k) log σ - |D x|²/(2σ²) + log N(Σx | 0, 0.001 n σ)` (R-INLA's
# `rw1`/`rw2`), checked in the values' own coordinates at two scales; the RK walk
# against its recurrence; RK = SB density and gradient at transported coordinates,
# with stated and known scales; SB forecasts continuing each law's recurrence;
# and the sparse Laplace integrating each walk whole: its value and gradient
# against the exact Gaussian integral for a Normal response (the order-2 prior
# improper in its trend included), and its gradient against central differences
# for a binomial response.
#
# RUN: julia --project=test test/rw_second_order.jl [testset-filter...]

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics, StanBlocks
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))

const BS = StanBlocks.BridgeStan
const RW2_CACHE = joinpath(tempdir(), "brm-rw-second-order")

# Ten yearly points; rows unsorted, two sharing a year.
const RW2_DATA = let rng = Xoshiro(20261120)
    t = Float64[3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 6]
    (; t, y=1.0 .+ 0.05 .* (t .- 5.5) .^ 2 .+ 0.2 .* randn(rng, length(t)))
end

# Daily binomial counts, as a rainfall series.
const RW2_COUNTS = let rng = Xoshiro(7)
    t = collect(1.0:12.0)
    trials = fill(20, length(t))
    prob = logistic.(-0.5 .+ 0.6 .* sin.(t ./ 2))
    (; t, trials, y=[rand(rng, Binomial(trials[i], prob[i])) for i in eachindex(t)])
end

const RW_BUILDERS = Dict(
    (order=1, sum_to_zero=false) => @brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + rw(t)
        y ~ Normal(mu, sigma)
    end),
    (order=2, sum_to_zero=false) => @brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + rw(t; order = 2)
        y ~ Normal(mu, sigma)
    end),
    (order=1, sum_to_zero=true) => @brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + rw(t; order = 1, sum_to_zero = true)
        y ~ Normal(mu, sigma)
    end),
    (order=2, sum_to_zero=true) => @brm(begin
        sigma ~ Exponential(1)
        mu ~ 1 + rw(t; order = 2, sum_to_zero = true)
        y ~ Normal(mu, sigma)
    end))
const RW_LAWS = sort!(collect(keys(RW_BUILDERS)); by=l -> (l.sum_to_zero, l.order))

stan(sb) = StanBlocks.stan_code(sb.model)
function instantiate(sb)
    code = stan(sb)
    mkpath(RW2_CACHE)
    StanBlocks.stan_instantiate(sb.model; path=joinpath(RW2_CACHE, string(hash(code)) * ".stan"))
end
function constrained(problem, q; seed=1)
    sm = problem.model
    names = String.(BS.param_names(sm; include_tp=true, include_gq=true))
    values = BS.param_constrain(sm, q; include_tp=true, include_gq=true,
        rng=BS.StanRNG(sm, seed))
    names, values
end
pick(names, values, prefix) =
    [v for (n, v) in zip(names, values) if n == prefix || startswith(n, prefix * ".")]

# The `order`-th differences of `n` values, as a matrix.
differences(n, order) = order == 1 ? diff(Matrix(1.0I, n, n); dims=1) :
    diff(diff(Matrix(1.0I, n, n); dims=1); dims=1)

@stestset "rw keywords: each law's program, and the refusals" begin
    data = RW2_DATA
    for law in RW_LAWS
        sb = SBBRMI(RW_BUILDERS[law](data); mod=@__MODULE__)
        code = stan(sb)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test occursin("rw_mu_t_sigma", code)
        if law.sum_to_zero
            @test occursin("brm_rw_sum", code)
            @test occursin("rw_mu_t_phi", code)
            @test sb.data[:rw_mu_t_order] == law.order
        else
            @test occursin(law.order == 1 ? "random_walk_path" : "random_walk2_rows", code)
            @test occursin("rw_mu_t_z", code)
            @test !haskey(sb.data, :rw_mu_t_order)
        end
    end
    # `rw(t)` is `rw(t; order = 1, sum_to_zero = false)`: the same program as before
    plain = SBBRMI(RW_BUILDERS[(order=1, sum_to_zero=false)](data); mod=@__MODULE__)
    spelled = SBBRMI(@brm(data, begin
        sigma ~ Exponential(1)
        mu ~ 1 + rw(t; order = 1, sum_to_zero = false)
        y ~ Normal(mu, sigma)
    end); mod=@__MODULE__)
    @test stan(plain) == stan(spelled)
    # `sd(lp, rw(t))` addresses the scale whatever the keywords
    stated = SBBRMI(@brm(data, begin
        mu ~ 1 + rw(t; order = 2, sum_to_zero = true)
        sd(:, rw(t)) ~ Normal(0.0, 0.05)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    @test occursin("rw_mu_t_sigma ~ normal(0.0, 0.05)", stan(stated))
    @test_throws "`order` must be the literal 1 or 2" SBBRMI(@brm(data, begin
        mu ~ 1 + rw(t; order = 3)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    @test_throws "unsupported keyword" SBBRMI(@brm(data, begin
        mu ~ 1 + rw(t; zero = :sum)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    @test_throws "needs at least 3 distinct" SBBRMI(@brm((; t=[1.0, 2.0, 2.0], y=zeros(3)), begin
        mu ~ 1 + rw(t; order = 2, sum_to_zero = true)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
end

@stestset "a sum-to-zero walk's values have the intrinsic law" begin
    steps = sort(unique(RW2_DATA.t)); n = length(steps)
    rng = Xoshiro(3)
    for order in (1, 2)
        rk = RKBRMI(RW_BUILDERS[(order=order, sum_to_zero=true)](RW2_DATA))
        names = coordinate_names(rk.model.layout)
        plan = RKEXT._rk_translated_plan(rk.plan)
        prior = prepare_query(rk.model, plan, :prior)
        jacobian = prepare_query(rk.model, plan, :log_jacobian)
        lp(u) = Base.invokelatest(prior, u) + Base.invokelatest(jacobian, u)
        is = findfirst(==(Symbol("rw_mu_t.sigma")), names)
        iphi = [findfirst(==(Symbol("rw_mu_t.phi.", k)), names) for k in 1:n]
        @test is !== nothing && all(!isnothing, iphi)
        D = differences(n, order)
        law(x, σ) = -(n - order) * log(σ) - sum(abs2, D * x) / (2σ^2) +
            logpdf(Normal(0, 0.001 * n * σ), sum(x))
        x = 0.3 .* randn(rng, n)
        x .-= mean(x) - 1e-4    # near the soft constraint, not on it
        point(σ) = (u = fill(0.1, length(names)); u[is] = log(σ); u[iphi] = x ./ σ; u)
        # The other coordinates agree, so the difference is the walk's law in its
        # values' coordinates (`phi = x / σ` adds `n log σ`), plus the scale's
        # `restricted(Normal(0, 0.2))` prior and its log Jacobian `log σ`.
        whole(σ) = law(x, σ) + n * log(σ) + logpdf(Normal(0, 0.2), σ) + log(σ)
        @test lp(point(1.9)) - lp(point(0.7)) ≈ whole(1.9) - whole(0.7) atol=1e-9
        @test lp(point(0.4)) - lp(point(0.7)) ≈ whole(0.4) - whole(0.7) atol=1e-9
    end
end

@stestset "RKBRMI: each walk's values equal its recurrence" begin
    steps = sort(unique(RW2_DATA.t)); n = length(steps)
    positions = [searchsortedfirst(steps, v) for v in RW2_DATA.t]
    for law in RW_LAWS
        rk = check_rk_source_roundtrip(RKBRMI(RW_BUILDERS[law](RW2_DATA)))
        names = coordinate_names(rk.model.layout)
        bound = rk_bound_data(rk)
        query = ReactiveKernels.prepare(rk.model.spec; have=(:unconstrained, keys(bound)...),
            want=(:rw_mu_t,), bound)
        is = findfirst(==(Symbol("rw_mu_t.sigma")), names)
        local_name = law.sum_to_zero ? "phi" : "z"
        iw = [findfirst(==(Symbol("rw_mu_t.", local_name, ".", k)), names)
            for k in 1:(law.sum_to_zero ? n : n - 1)]
        @test all(!isnothing, iw)
        for u in (collect(range(-0.6, 0.7; length=length(names))),
                  0.3 .* randn(Xoshiro(8), length(names)))
            σ, w = exp(u[is]), u[iw]
            path = law.sum_to_zero ? σ .* w :
                law.order == 1 ? vcat(0.0, σ .* cumsum(w)) : vcat(0.0, cumsum(σ .* cumsum(w)))
            value = Base.invokelatest(query, u)
            @test (value isa Tuple ? only(value) : value) ≈ path[positions] atol=1e-12 rtol=1e-12
        end
    end
end

@stestset "RKBRMI: the same law as SBBRMI at transported coordinates" begin
    data = RW2_DATA
    stated = @brm(data, begin
        sigma ~ Exponential(1)
        mu ~ 1 + rw(t; order = 2, sum_to_zero = true)
        sd(mu, rw(t)) ~ Exponential(2)
        y ~ Normal(mu, sigma)
    end)
    known = @brm(data, begin
        sigma ~ Exponential(1)
        mu ~ 1 + rw(t; order = 2)
        sd(mu, rw(t)) = 0.3
        y ~ Normal(mu, sigma)
    end)
    for brmi in (map(law -> RW_BUILDERS[law](data), RW_LAWS)..., stated, known)
        rk = RKBRMI(brmi)
        problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        sb = SBBRMI(brmi; mod=@__MODULE__)
        stan_problem = instantiate(sb)
        stan_names = BS.param_unc_names(stan_problem.model)
        transport = brm_coordinate_transport(rk, sb, stan_names)
        @test length(transport) == length(coordinate_names(rk.model.layout)) ==
            length(stan_names)
        for u in (zeros(length(transport)),
                  collect(range(-0.43, 0.51; length=length(transport))))
            stan_u = brm_rk_point_to_stan(transport, u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_value, stan_gradient = BS.log_density_gradient!(stan_problem.model, stan_u,
                zeros(length(stan_u)); propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=1e-9 rtol=1e-12
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=1e-8 rtol=1e-10
        end
    end
end

@stestset "SBBRMI forecasts continue each law's recurrence" begin
    rows(times; with_y=true) = with_y ?
        (; t=collect(Float64, times), y=0.1 .* cos.(1:length(times))) :
        (; t=collect(Float64, times))
    for law in RW_LAWS
        law == (order=1, sum_to_zero=false) && continue   # test/walk_forecast.jl
        sb = SBBRMI(RW_BUILDERS[law](rows(1:5)); mod=@__MODULE__)
        @test sb.data[:rw_mu_t_n_new_steps] == 0
        fc = reprocess(sb, rows(5:7; with_y=false))
        @test fc.data[:rw_mu_t_n_steps] == 5
        @test StanBlocks.getvalue(fc.data[:rw_mu_t_n_new_steps]) == 2
        fit_problem, fc_problem = instantiate(sb), instantiate(fc)
        dimension = LogDensityProblems.dimension(fit_problem)
        @test LogDensityProblems.dimension(fc_problem) == dimension
        q = 0.4 .* randn(Xoshiro(5), dimension)
        fn, fv = constrained(fit_problem, q)
        σ = only(pick(fn, fv, "rw_mu_t_sigma"))
        fitted = pick(fn, fv, "rw_mu_t")
        @test length(fitted) == 5
        law.sum_to_zero && @test abs(sum(fitted)) < 0.01 * σ * 5 + 1   # soft, not exact
        for seed in 1:20
            cn, cv = constrained(fc_problem, q; seed)
            path = pick(cn, cv, "rw_mu_t")
            z_new = pick(cn, cv, "rw_mu_t_z_new")
            @test length(z_new) == 2
            @test path[1] ≈ fitted[5] atol=1e-10
            if law.order == 1
                @test path[2] ≈ fitted[5] + σ * z_new[1] atol=1e-10
                @test path[3] ≈ path[2] + σ * z_new[2] atol=1e-10
            else
                @test path[2] ≈ 2fitted[5] - fitted[4] + σ * z_new[1] atol=1e-10
                @test path[3] ≈ 2path[2] - fitted[5] + σ * z_new[2] atol=1e-10
            end
        end
    end
end

@stestset "NLME view: a walk is shared by the units" begin
    data = (; RW2_DATA..., subject=["s$(i % 3 + 1)" for i in eachindex(RW2_DATA.t)])
    for brmi in (@brm(data, begin
                sigma ~ Exponential(1)
                mu ~ 1 + rw(t; order = 2, sum_to_zero = true) + (1 | subject)
                y ~ Normal(mu, sigma)
            end), @brm(data, begin
                sigma ~ Exponential(1)
                mu ~ 1 + rw(t; order = 2) + (1 | subject)
                y ~ Normal(mu, sigma)
            end))
        rk = RKBRMI(brmi)
        v = brm_nlme_view(rk; subject_group=:subject)
        @test !any(==(:unclaimed), v.role)
        walk = [i for (i, name) in enumerate(v.coordinates) if startswith(string(name), "rw_mu_t.")]
        @test !isempty(walk)
        @test all(==(:shared_latent), v.role[walk])
    end
end

spl_point(p, φ) = (u = copy(p.u); u[p.population] .= φ; u[p.integrated] .= 0.0; u)
function spl_fd(f, φ; h=1e-5)
    map(eachindex(φ)) do k
        e = zeros(length(φ)); e[k] = h
        (f(φ .+ e) - f(φ .- e)) / 2h
    end
end

@stestset "sparse Laplace: each walk integrated whole, the exact Gaussian integral" begin
    data = RW2_DATA
    steps = sort(unique(data.t)); n = length(steps)
    A = [Float64(data.t[i] == steps[j]) for i in eachindex(data.t), j in 1:n]
    rng = Xoshiro(11)
    for law in RW_LAWS
        rk = RKBRMI(RW_BUILDERS[law](data))
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:t,))
        @test length(p.integrated) == (law.sum_to_zero ? n : n - 1)
        @test length(p.components) == 1
        # The integrated draws' prior precision and their map to the walk's values
        # at unit scale: the standardized values under `DᵀD` plus the soft sum, or
        # the innovations, standard normal, summed once or twice.
        M0, P = if law.sum_to_zero
            D = differences(n, law.order)
            A, D' * D .+ 1 / (0.001n)^2
        else
            C = Matrix(LowerTriangular(ones(n - 1, n - 1)))
            A * vcat(zeros(1, n - 1), law.order == 1 ? C : C * C), Matrix(1.0I, n - 1, n - 1)
        end
        plan = RKEXT._rk_translated_plan(rk.plan)
        prior = prepare_query(rk.model, plan, :prior)
        jacobian = prepare_query(rk.model, plan, :log_jacobian)
        # ∫ exp(-½ zᵀPz) N(y | μ₀ + τ M₀ z, diag(σ²)) dz, with RK's prior at z = 0:
        # exact for a Normal response, also where P is singular (the order-2
        # trend), since the response identifies that direction.
        function oracle(φ)
            u = spl_point(p, φ)
            ψ = BRM._brm_slap_psi!(zeros(length(p.port)), p, BRM._rk_slap_values(p.kernels, u))
            μ0 = [ψ[p.entry[t]] for t in eachindex(p.family)]
            σ = [exp(ψ[p.entry[t] + 1]) for t in eachindex(p.family)]
            τ = only(getproperty(BRM._rk_constrained_values(rk, u),
                p.blocks[1].declaration).sigma)
            M = τ .* M0
            W = Diagonal(1 ./ σ .^ 2)
            r = data.y .- μ0
            H = Symmetric(P + M' * W * M)
            b = M' * W * r
            Base.invokelatest(prior, u) + Base.invokelatest(jacobian, u) -
                r' * W * r / 2 + b' * (H \ b) / 2 - logdet(H) / 2 +
                (size(P, 1) - length(r)) / 2 * log(2π) - sum(log, σ)
        end
        for _ in 1:2
            φ = 0.3 .* randn(rng, length(p.population))
            value, gradient = LogDensityProblems.logdensity_and_gradient(p, φ)
            @test value ≈ oracle(φ) atol=1e-7
            @test gradient ≈ spl_fd(oracle, φ) rtol=1e-6 atol=1e-6
        end
    end
    # Counts, as R-INLA's binomial `rw2` example: the gradient of the Laplace
    # marginal itself, where the curvature moves with the mode.
    for law in ((order=2, sum_to_zero=true), (order=2, sum_to_zero=false))
        brmi = law.sum_to_zero ? @brm(RW2_COUNTS, begin
                logit(prob) ~ 1 + rw(t; order = 2, sum_to_zero = true)
                y ~ Binomial(trials, prob)
            end) : @brm(RW2_COUNTS, begin
                logit(prob) ~ 1 + rw(t; order = 2)
                y ~ Binomial(trials, prob)
            end)
        q = brm_laplace_problem(RKBRMI(brmi); ad_backend=NLME_AD, integrate=(:t,))
        φ = 0.3 .* randn(rng, length(q.population))
        value, gradient = LogDensityProblems.logdensity_and_gradient(q, φ)
        @test isfinite(value)
        @test gradient ≈ spl_fd(ψ -> LogDensityProblems.logdensity(q, ψ), φ) rtol=1e-6 atol=1e-6
    end
end

println("rw_second_order.jl: all testsets passed")
