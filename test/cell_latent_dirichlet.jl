# test/cell_latent_dirichlet.jl — a simplex per `@plate` cell:
# `theta ~ Dirichlet(K, a)` or `theta ~ Dirichlet(alpha)` on a bare cell name
# (snag `mixed-membership-1a1930a6`, design decision `0b9jgl5`).
#
# The latent is sampled in the K - 1 coordinates of `shares(total)` with the
# total 1, `theta = softmax(sum_to_zero_constrain(y))`, and the density of the
# coordinates is the Dirichlet's at `theta` plus `sum(log.(theta))`. Checked:
# that density against an independent reading at RK points, RKBRMI against
# SBBRMI through the coordinate transport (value and gradient), the symmetric,
# vector and sampled concentrations, K = 2 and 3, starts, draws, and SVI giving
# each cell's simplex to the unit whose rows it reads.
#
# Run: julia --startup-file=no --project=test test/cell_latent_dirichlet.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, LinearAlgebra

# Long-format rows: S series of T rows, one simplex per series.
function mixing_data(; S=4, T=5, K=2, seed=1)
    rng = Xoshiro(seed)
    series = repeat(1:S, inner=T)
    N = length(series)
    X = round.(randn(rng, N, K); digits=3)
    (; y=round.(randn(rng, N); digits=3), x1=X[:, 1], x2=X[:, 2],
        x3=K >= 3 ? X[:, 3] : zeros(N), series, n=K, alpha0=1.5,
        alpha=collect(range(0.8, 2.0; length=K)))
end
const D2 = mixing_data()
const D3 = mixing_data(; K=3, seed=2)
# One concentration per cell, in a plate over the series.
const D3C = merge(D3, (; labels=collect(1:4),
    alpha_cells=[[1.0, 1.5, 2.0], [0.5, 0.5, 3.0], [2.0, 2.0, 2.0], [1.2, 0.7, 0.9]]))

symmetric(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for p in groups(series)
        theta ~ Dirichlet(n, alpha0)
        y[p] ~ Normal(theta[1] .* x1[p] .+ theta[2] .* x2[p], sigma)
    end
end
concentration_vector(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for p in groups(series)
        theta ~ Dirichlet(alpha)
        y[p] ~ Normal(theta[1] .* x1[p] .+ theta[2] .* x2[p] .+ theta[3] .* x3[p], sigma)
    end
end
per_cell(data) = @brm data begin
    sigma ~ Exponential(1.0)
    rows = groups(series => labels)
    @plate for i in eachindex(alpha_cells)
        theta ~ Dirichlet(alpha_cells[i])
        y[rows[i]] ~ Normal(theta[1] .* x1[rows[i]] .+ theta[2] .* x2[rows[i]] .+
            theta[3] .* x3[rows[i]], sigma)
    end
end
# A sampled concentration: the Dirichlet's normalizing constant reads it.
sampled(data) = @brm data begin
    sigma ~ Exponential(1.0)
    a ~ LogNormal(0.0, 0.5)
    @plate for p in groups(series)
        theta ~ Dirichlet(n, a)
        y[p] ~ Normal(theta[1] .* x1[p] .+ theta[2] .* x2[p], sigma)
    end
end

const AD = AutoEnzyme(; mode=Enzyme.Reverse)
softmax_values(y) = (z = BRM._brm_sum_to_zero_values(y); w = exp.(z .- maximum(z)); w ./ sum(w))

# The model's log density at the RK point `u`, read independently: the priors
# with their log transforms' Jacobians, each cell's Dirichlet at its simplex
# plus `sum(log.(theta))`, and the observations.
# `coordinate(s, k)` reads cell `s`'s k-th simplex coordinate, `scalar(name)` a
# scalar's unconstrained value (`nothing` when the model has none).
function oracle(data, coordinate, scalar; concentration)
    sigma = exp(scalar(:sigma))
    lp = logpdf(Exponential(1.0), sigma) + log(sigma)
    a = nothing
    if scalar(:a) !== nothing
        a = exp(scalar(:a))
        lp += logpdf(LogNormal(0.0, 0.5), a) + log(a)
    end
    K = data.n
    X = hcat(data.x1, data.x2, data.x3)[:, 1:K]
    for s in sort!(unique(data.series))
        y = [coordinate(s, k) for k in 1:K-1]
        theta = softmax_values(y)
        alpha = concentration === :vector ? data.alpha :
            concentration === :per_cell ? data.alpha_cells[s] :
            fill(concentration === :sampled ? a : data.alpha0, K)
        lp += logpdf(Dirichlet(alpha), theta) + sum(log.(theta))
        rows = findall(==(s), data.series)
        lp += sum(logpdf.(Normal.(X[rows, :] * theta, sigma), data.y[rows]))
    end
    lp
end

function rk_oracle(rk, data, u; concentration)
    names = Symbol.(coordinate_names(rk.model.layout))
    at(name) = (j = findfirst(==(name), names); j === nothing ? nothing : u[j])
    oracle(data, (s, k) -> at(Symbol("theta.", s, ".", k)), at; concentration)
end
function stan_oracle(stan, data, u; concentration)
    names = String.(BridgeStan.param_unc_names(stan.model))
    at(name) = (j = findfirst(==(name), names); j === nothing ? nothing : u[j])
    coordinate(s, k) = u[only(findall(n -> endswith(n, "theta_coordinates.$k.$s"), names))]
    oracle(data, coordinate, n -> at(String(n)); concentration)
end

function check_model(brmi, data, label; concentration, points=3)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(check_rk_source_roundtrip(rk); ad_backend=AD)
    names = Symbol.(coordinate_names(rk.model.layout))
    S, K = length(unique(data.series)), data.n
    @test count(n -> startswith(String(n), "theta."), names) == S * (K - 1)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    folder = joinpath(tempdir(), "brm-cell-dirichlet")
    mkpath(folder)
    stan = BRM.stan_instantiate(sb; path=joinpath(folder, label * ".stan"))
    t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
    rng = Xoshiro(17)
    d = LogDensityProblems.dimension(problem)
    for _ in 1:points
        u = 0.4 .* randn(rng, d)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test value ≈ rk_oracle(rk, data, u; concentration) atol=1e-10 rtol=1e-11
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model,
            brm_rk_to_stan(t, u); propto=false, jacobian=true)
        @test value ≈ stan_value atol=1e-10 rtol=1e-11
        @test gradient ≈ brm_stan_to_rk(t, stan_gradient) atol=1e-9 rtol=1e-10
        @test brm_check_coordinate_transport(t, rk, stan.model, u) isa NamedTuple
    end
    (; rk, sb, stan, t, problem)
end

@stestset "symmetric Dirichlet(n, a): density, and RKBRMI against SBBRMI" begin
    check_model(symmetric(D2), D2, "symmetric"; concentration=:symmetric)
end

@stestset "Dirichlet(alpha) per cell with K = 3" begin
    check_model(per_cell(D3C), D3C, "per_cell"; concentration=:per_cell)
end

@stestset "a concentration vector shared by every cell: SBBRMI; RKBRMI refuses it" begin
    # RKPPL binds the shared data vector as a per-index column (ReactiveKernels
    # snag rkppl-joint-cell-c8a61b47), so RKBRMI refuses it by name.
    @test_throws "rkppl-joint-cell-c8a61b47" RKBRMI(concentration_vector(D3))
    sb = SBBRMI(concentration_vector(D3); mod=@__MODULE__, total_groups=())
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "brm-cell-dirichlet", "shared.stan"))
    rng = Xoshiro(5)
    d = BridgeStan.param_unc_num(stan.model)
    for _ in 1:3
        u = 0.4 .* randn(rng, d)
        @test BridgeStan.log_density(stan.model, u; propto=false, jacobian=true) ≈
            stan_oracle(stan, D3, u; concentration=:vector) atol=1e-10 rtol=1e-11
    end
end

@stestset "a sampled concentration" begin
    check_model(sampled(D2), D2, "sampled"; concentration=:sampled)
end

@stestset "draws and starts read the simplex by its name" begin
    rk = RKBRMI(symmetric(D2))
    names = Symbol.(coordinate_names(rk.model.layout))
    U = 0.5 .* randn(Xoshiro(3), length(names), 4)
    draws = rk_restore_draws(rk, U)
    S = length(unique(D2.series))
    @test length(draws.theta) == S
    for s in 1:S
        @test size(draws.theta[s]) == (2, 4)
        @test all(x -> isapprox(x, 1; atol=1e-12), sum(draws.theta[s]; dims=1))
        @test all(>(0), draws.theta[s])
        y = U[findfirst(==(Symbol("theta.", s, ".1")), names), :]
        @test draws.theta[s][:, 2] ≈ softmax_values([y[2]]) rtol=1e-12
    end
    theta = [[0.3, 0.7], [0.5, 0.5], [0.9, 0.1], [0.25, 0.75]]
    start = brm_start(rk, (; theta, sigma=0.5))
    u = start isa NamedTuple ? start.u : start
    r = rk_restore_draws(rk, reshape(u, :, 1))
    @test [vec(r.theta[s]) for s in 1:S] ≈ theta rtol=1e-10
    @test only(r.sigma) ≈ 0.5 rtol=1e-12
    @test_throws "sum to 1" brm_start(rk, (; theta=[[0.3, 0.6], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8]],
        sigma=0.5))
    # SBBRMI: the same start, through the cell's coordinates `theta_coordinates`.
    sb = SBBRMI(symmetric(D2); mod=@__MODULE__, total_groups=())
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "brm-cell-dirichlet", "start.stan"))
    v = brm_start(sb, stan.model, (; theta, sigma=0.5))
    values = BridgeStan.param_constrain(stan.model, v; include_tp=true)
    names = String.(BridgeStan.param_names(stan.model; include_tp=true))
    at(k, s) = values[only(findall(n -> endswith(n, "_theta.$k.$s"), names))]
    @test [[at(k, s) for k in 1:2] for s in 1:S] ≈ theta rtol=1e-10
    @test_throws "sum to 1" brm_start(sb, stan.model,
        (; theta=[[0.3, 0.6], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8]], sigma=0.5))
end

# The reporter's shape: a level per series beside its simplex.
with_levels(data) = @brm data begin
    sigma ~ Exponential(1.0)
    mu ~ 0 + (1 | series)
    @plate for p in groups(series)
        theta ~ Dirichlet(n, alpha0)
        y[p] ~ Normal(mu[p] .+ theta[1] .* x1[p] .+ theta[2] .* x2[p], sigma)
    end
end

@stestset "SVI fits the simplexes; each is still a global of the fit" begin
    rk = RKBRMI(with_levels(D2))
    m = brm_nlme_model(rk; ad_backend=AD, subject_group=:series)
    t = BRM._brm_svi_target(m)
    names = string.(m.view.coordinates)
    for (j, name) in enumerate(names)
        startswith(name, "theta.") || continue
        series = parse(Int, split(name, '.')[2])
        # Every unit's cell footprint reads every cell's simplex block, so SVI
        # keeps them global (ReactiveKernels snag joint-cell-laten-9a3ad461). The
        # authored factors are per cell already.
        @test_broken t.ownership.owner[j] == findfirst(==(series), m.view.levels)
        @test t.ownership.owner[j] == 0
    end
    @test count(n -> startswith(n, "theta."), names) == length(unique(D2.series))
    fit = brm_svi_fit(m; batch_size=2, max_iterations=30, rng=Xoshiro(9))
    @test size(fit.draws, 1) == length(names)
    @test all(isfinite, fit.draws)
end

@stestset "refusals name the spelling" begin
    message(f) = try f(); "" catch e; sprint(showerror, e) end
    @test occursin("positional arguments", message(() -> @eval @brm begin
        @plate for p in groups(series)
            theta ~ Dirichlet(alpha; check_args=false)
            y[p] ~ Normal(theta[1], 1.0)
        end
    end))
end
