# test/cell_latent_total_shares.jl — `@plate` cell latents with a given total:
# `T::shares(total) ~ Law(…)`.
#
# A positive latent that sums to `total` in every draw, with `Law` as its law
# on that surface. The sampler moves the K - 1 coordinates of Stan's
# `simplex[K]` (the shares `T ./ total`), and the density of the coordinates is
# the law's at `T` plus `sum(log.(T)) - log(total)` (snag
# exact-sum-constr-7ba55c76, user decisions 18wcqyi, 10kbiuy and 1a5fvjo).
#
# Run: julia --startup-file=no --project=test test/cell_latent_total_shares.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, LinearAlgebra

# Cells of 3, 2 and 1 entries (one entry: the latent is its total).
const CELLS = (; mu=[[10.0, 20.0, 5.0], [7.0, 3.0], [4.0]], obs=[[1, 2], [1], [1]],
    y=[[11.0, 19.0], [6.5], [4.2]], total=[36.0, 10.5, 4.1], frac=[1.0, 10.5 / 36, 4.1 / 36])

shares_lognormal(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        T::shares(total[i]) ~ LogNormal(log.(mu[i]), 0.5)
        y[i] ~ Normal(T[obs[i]], sigma)
    end
end
# A scale per entry that reads a sampled scalar: the law's location alone
# sizes the latent on RKBRMI.
shares_scale(data) = @brm data begin
    sigma ~ Exponential(1.0)
    s ~ Exponential(2.0)
    @plate for i in eachindex(total)
        loc = log.(mu[i])
        scl = s .+ 0.1 .* loc
        T::shares(total[i]) ~ LogNormal(loc, scl)
        y[i] ~ Normal(T[obs[i]], sigma)
    end
end
# Another positive family, sized by its second argument.
shares_gamma(data) = @brm data begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(total)
        T::shares(total[i]) ~ Gamma(4.0, mu[i] ./ 4.0)
        y[i] ~ Normal(T[obs[i]], sigma)
    end
end
# A total that is a model value: a sampled scale times data.
shares_sampled_total(data) = @brm data begin
    sigma ~ Exponential(1.0)
    tau ~ LogNormal(log(36.0), 0.1)
    @plate for i in eachindex(total)
        T::shares(tau * frac[i]) ~ LogNormal(log.(mu[i]), 0.5)
        y[i] ~ Normal(T[obs[i]], sigma)
    end
end
# One cell of three, and Stan's own simplex program for it.
const ONE = (; mu=[[10.0, 20.0, 5.0]], obs=[[1, 2]], y=[[11.0, 19.0]], total=[36.0])
one_cell(data) = @brm data begin
    @plate for i in eachindex(total)
        T::shares(total[i]) ~ LogNormal(log.(mu[i]), 0.5)
        y[i] ~ Normal(T[obs[i]], 1.0)
    end
end
const STAN_SIMPLEX = """
data { vector[3] mu; real<lower=0> total; vector[2] y; array[2] int obs; }
parameters { simplex[3] w; }
transformed parameters { vector[3] T = total * w; }
model { T ~ lognormal(log(mu), 0.5); y ~ normal(T[obs], 1.0); }
"""

function compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    folder = joinpath(tempdir(), "brm-total-shares")
    mkpath(folder)
    stan = BRM.stan_instantiate(sb; path=joinpath(folder, label * ".stan"))
    (; sb, stan, names=BridgeStan.param_unc_names(stan.model))
end

# The constrained values of `T` at an unconstrained point, one vector per cell.
function cell_values(c, u, data)
    θ = BridgeStan.param_constrain(c.stan.model, u; include_tp=true)
    names = BridgeStan.param_names(c.stan.model; include_tp=true)
    T = θ[findall(n -> occursin(r"_T\.\d+$", n), names)]
    ends = cumsum(length.(data.mu))
    [T[e-length(m)+1:e] for (e, m) in zip(ends, data.mu)]
end
named(c, u, name) = BridgeStan.param_constrain(c.stan.model, u; include_tp=true)[
    findfirst(==(name), BridgeStan.param_names(c.stan.model; include_tp=true))]

# The log density an independent reading of the model gives at `T` (as Stan
# constrains it): the priors with their log transforms' Jacobians, the law at
# each cell's `T` plus `sum(log.(T)) - log(total)`, and the observations.
function oracle(T, sigma, totals, law; tau=nothing)
    lp = logpdf(Exponential(1.0), sigma) + log(sigma)
    tau === nothing || (lp += logpdf(LogNormal(log(36.0), 0.1), tau) + log(tau))
    for (c, t) in enumerate(T)
        lp += sum(logpdf.(law(c), t)) + sum(log.(t)) - log(totals[c])
        lp += sum(logpdf.(Normal.(t[CELLS.obs[c]], sigma), CELLS.y[c]))
    end
    lp
end

# RKBRMI's value and native Enzyme gradient against SBBRMI's through the
# coordinate transport, and the printed RK source replays.
function check_against_stan(brmi, name; points=3, scale=0.3)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(check_rk_source_roundtrip(rk);
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "brm-total-shares", name * ".stan"))
    t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(stan.model)))
    rng = Xoshiro(17)
    n = LogDensityProblems.dimension(problem)
    for _ in 1:points
        u = scale .* randn(rng, n)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model,
            brm_rk_to_stan(t, u); propto=false, jacobian=true)
        @test value ≈ stan_value atol=1e-10 rtol=1e-11
        @test gradient ≈ brm_stan_to_rk(t, stan_gradient) atol=1e-9 rtol=1e-10
        @test brm_check_coordinate_transport(t, rk, stan.model, u) isa NamedTuple
    end
    (; rk, sb, stan, t, problem)
end

@stestset "Julia maps: the values sum to the total, and back" begin
    rng = Xoshiro(1)
    for K in (1, 2, 5, 12)
        T = exp.(2 .* randn(rng, K))
        y = BRM._brm_shares_coordinates(T)
        @test length(y) == K - 1
        @test BRM._brm_shares_values(y, sum(T)) ≈ T rtol=1e-12
        @test sum(BRM._brm_shares_values(randn(rng, K - 1), 7.5)) ≈ 7.5 rtol=1e-12
    end
    @test BRM._brm_shares_values(Float64[], 4.1) ≈ [4.1]
end

@stestset "SBBRMI: Stan's own simplex program, up to a constant" begin
    shares = compiled(one_cell(ONE), "one_cell")
    stan = BridgeStan.StanModel(let path = joinpath(tempdir(), "brm-total-shares", "stan_simplex.stan")
            write(path, STAN_SIMPLEX); path
        end, """{"mu": [10.0, 20.0, 5.0], "total": 36.0, "y": [11.0, 19.0], "obs": [1, 2]}"""; warn=false)
    @test length(shares.names) == 2 == BridgeStan.param_unc_num(stan)
    rng = Xoshiro(2)
    diffs = map(1:8) do _
        y = randn(rng, 2)
        T = cell_values(shares, y, ONE)[1]
        @test sum(T) ≈ 36.0 rtol=1e-12
        @test T ≈ BridgeStan.param_constrain(stan, y; include_tp=true)[4:6] rtol=1e-12
        BridgeStan.log_density(shares.stan.model, y; propto=false) -
            BridgeStan.log_density(stan, y; propto=false)
    end
    # Ours adds sum(log T) - log(total); Stan's simplex adds sum(log w) + log(3) / 2.
    @test maximum(diffs) - minimum(diffs) < 1e-10
    @test diffs[1] ≈ 2 * log(36.0) - log(3) / 2 atol=1e-10
end

@stestset "SBBRMI: the density at the constrained values, every cell at its total" begin
    for (model, label, law) in (
            (shares_lognormal, "lognormal", c -> LogNormal.(log.(CELLS.mu[c]), 0.5)),
            (shares_gamma, "gamma", c -> Gamma.(4.0, CELLS.mu[c] ./ 4.0)))
        shares = compiled(model(CELLS), label)
        @test length(shares.names) == 1 + 3
        rng = Xoshiro(3)
        for _ in 1:8
            u = 0.5 .* randn(rng, length(shares.names))
            T = cell_values(shares, u, CELLS)
            @test sum.(T) ≈ CELLS.total rtol=1e-12
            @test BridgeStan.log_density(shares.stan.model, u; propto=false) ≈
                oracle(T, named(shares, u, "sigma"), CELLS.total, law) atol=1e-8 rtol=1e-12
            @test all(isfinite, last(BridgeStan.log_density_gradient(shares.stan.model, u; propto=false)))
        end
    end
    # A total that is a model value: the density reads it.
    shares = compiled(shares_sampled_total(CELLS), "sampled_total")
    rng = Xoshiro(4)
    for _ in 1:8
        u = 0.3 .* randn(rng, length(shares.names))
        T = cell_values(shares, u, CELLS)
        tau = named(shares, u, "tau")
        @test sum.(T) ≈ tau .* CELLS.frac rtol=1e-12
        @test BridgeStan.log_density(shares.stan.model, u; propto=false) ≈
            oracle(T, named(shares, u, "sigma"), tau .* CELLS.frac,
                c -> LogNormal.(log.(CELLS.mu[c]), 0.5); tau) atol=1e-8 rtol=1e-12
    end
    d = brm_descriptor(SBBRMI(shares_lognormal(CELLS); mod=@__MODULE__, total_groups=()))
    @test brm_output(d, :T_coordinates).segments == [2, 3, 3]
    @test !isempty(brm_outputs(d; logical=:T))
end

@stestset "RKBRMI agrees with SBBRMI: value, gradient, transport" begin
    for (model, label) in ((shares_lognormal, "rk_lognormal"), (shares_scale, "rk_scale"),
            (shares_gamma, "rk_gamma"), (shares_sampled_total, "rk_sampled_total"))
        check_against_stan(model(CELLS), label)
    end
end

@stestset "RKBRMI declares one joint draw per cell" begin
    rk = RKBRMI(shares_lognormal(CELLS))
    source = string(BRM._rk_retained_program(rk))
    @test occursin("T[i] ~ ", source)
    @test !occursin("T[i] .~", source)
    @test coordinate_names(rk.model.layout)[2:end] == Symbol.(["T.1.1", "T.1.2", "T.2.1"])
    # Draws restore one entries × draws matrix per cell, as values at the totals.
    U = randn(Xoshiro(5), 4, 3)
    restored = rk_restore_draws(rk, U)
    @test length(restored.T) == 3
    @test vec(sum(restored.T[1]; dims=1)) ≈ fill(36.0, 3) rtol=1e-12
    @test restored.T[1] ≈ reduce(hcat, BRM._brm_shares_values(U[2:3, k], 36.0) for k in 1:3)
    @test restored.T[3] ≈ fill(4.1, 1, 3)
end

@stestset "brm_start takes the latent's values on both backends" begin
    brmi = shares_lognormal(CELLS)
    rk = RKBRMI(brmi)
    shares = compiled(brmi, "start_lognormal")
    t = brm_coordinate_transport(rk, shares.sb, String.(shares.names))
    values = [[9.0, 21.0, 6.0], [7.5, 3.0], [4.1]]
    for given in (values, reduce(vcat, values))
        u = brm_start(shares.sb, shares.stan.model, (; T=given, sigma=0.5))
        @test cell_values(shares, u, CELLS) ≈ values rtol=1e-10
        v = brm_start(rk, (; T=given, sigma=0.5))
        @test brm_rk_to_stan(t, v) ≈ u rtol=1e-12 atol=1e-14
        @test BRM._rk_constrained_values(rk, v).T ≈ values rtol=1e-12
    end
    @test_throws DimensionMismatch brm_start(shares.sb, shares.stan.model, (; T=values[1:2]))
    # refused: values that miss a cell's total are no point of the latent, and
    # are not rescaled onto it.
    off = [[9.0, 21.0, 7.0], [7.5, 3.0], [4.1]]
    @test_throws "must sum to each cell's total" brm_start(shares.sb, shares.stan.model, (; T=off))
    @test_throws "must sum to its total" brm_start(rk, (; T=off))
    # refused: a positive latent's values must be positive.
    @test_throws "positive values" brm_start(shares.sb, shares.stan.model,
        (; T=[[9.0, -1.0, 28.0], [7.5, 3.0], [4.1]]))
end

@stestset "refused spellings" begin
    # refused: `shares(total)` is the bijection a typed left-hand side names
    # (decision 1a5fvjo, which replaced `total_shares`).
    @test_throws "names no bijection" macroexpand(@__MODULE__, :(@brm CELLS begin
        @plate for i in eachindex(total)
            T::total_shares ~ LogNormal(log.(mu[i]), 0.5)
            y[i] ~ Normal(T[obs[i]], 1.0)
        end
    end))
    # refused: the bijection needs the total the latent sums to.
    @test_throws "takes the total" macroexpand(@__MODULE__, :(@brm CELLS begin
        @plate for i in eachindex(total)
            T::shares ~ LogNormal(log.(mu[i]), 0.5)
            y[i] ~ Normal(T[obs[i]], 1.0)
        end
    end))
    # refused: a typed left-hand side declares a latent, a bare cell name.
    @test_throws "not a bare name" macroexpand(@__MODULE__, :(@brm CELLS begin
        @plate for i in eachindex(total)
            T[1]::shares(total[i]) ~ LogNormal(log.(mu[i]), 0.5)
            y[i] ~ Normal(T[obs[i]], 1.0)
        end
    end))
    # refused: `shares` takes a positive latent; a log-scale latent is the same
    # model written as `T ~ LogNormal(…)`.
    for build in (brmi -> SBBRMI(brmi; mod=@__MODULE__, total_groups=()), RKBRMI)
        @test_throws "declares a positive latent" build(@brm CELLS begin
            @plate for i in eachindex(total)
                eta::shares(total[i]) ~ Normal(log.(mu[i]), 0.5)
                y[i] ~ Normal(exp.(eta)[obs[i]], 1.0)
            end
        end)
    end
end

@stestset "descriptor, plan and description read the model" begin
    sb = SBBRMI(shares_lognormal(CELLS); mod=@__MODULE__, total_groups=())
    d = brm_descriptor(sb)
    plan = generative_plan(sb)
    @test any(decl -> occursin("T_coordinates", string(decl.target)), plan.declarations)
    text = string(brm_description(d))
    @test occursin("T", text)
end
