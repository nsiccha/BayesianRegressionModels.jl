# test/adaptive_centering_location.jl — hierarchical centring in adaptive
# centring, and the opt-in `centeredness = :midpoint` start.
#
# A cell of a block compiled non-centred is centred on its predictor's matching
# population coefficient, so its `c = 1` coordinate is the level's total
# `beta + b[j]` (snag fitted-seven-hie-76d07973, decision
# `BayesianRegressionModels:snag.fitted-seven-hie-76d07973/decisions/2026-10-26T13-20-57-865-1xdxiv1`).
# The oracles are independent of the wrapper's accessors:
#
# - which coefficient each cell is centred on, from the formula: intercept and
#   slope terms of a top-level grouping take theirs; a nested grouping's level
#   takes the intercept plus its enclosing level's effect (snag
#   brm-zone-8-adapt-1b407672; `test/adaptive_centering_total_location.jl`
#   covers factor levels, chains and crossed groupings); a term without a
#   population coefficient, a block compiled centred and a predictor whose
#   coefficient S2Z absorbed take none;
# - the map in closed form, `z = (x - c * beta) / s^c` per K = 1 cell and the
#   triangular form for a correlated block, with scales and the correlation
#   factor read from BridgeStan's constrained values;
# - the wrapper's log density and gradient against central finite differences
#   of its own log density, on every wrapper path that carries a location: the
#   hand-written K <= 2 pullback, the block-level pass (K = 3), the joint
#   derived-scale wrapper (Student-t) and the learned order;
# - `:midpoint` starts random-effect cells at 0.5 and HSGP cells at their
#   compiled frame.
#
# Run: julia --project=test test/adaptive_centering_location.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LinearAlgebra
using LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, Normal, TDist

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan

instantiate(model) = Base.invokelatest(StanBlocks.stan_instantiate, model.model)

rng = Xoshiro(7)
G = 12
g = repeat(["g$j" for j in 1:G]; inner = 6)
band = ["b$(cld(parse(Int, s[2:end]), 4))" for s in g]   # 3 bands, each holding 4 g-levels
h = repeat(["h$j" for j in 1:3]; outer = 24)             # crossed with g
x, t, y = randn(rng, 72), randn(rng, 72), randn(rng, 72)
data = (; g, band, h, x, t, y)

# Correlated (1 + x | g), crossed (1 | h), and a slope on `t`, which has no
# population coefficient.
const M_MAIN = SBBRMI(merge(@brm(begin
    mu ~ 1 + x + (1 + x | g) + (1 | h) + (0 + t | h)
    y ~ Normal(mu, sigma)
end), @brm(begin
    sigma ~ Exponential(1)
end))(data); mod = @__MODULE__, total_groups = ())
const P_MAIN = instantiate(M_MAIN)
const UNC_MAIN = BS.param_unc_names(P_MAIN.model)

coordinate(name) = only(findall(==(name), UNC_MAIN))
intercept_name(unc) = only(n for n in unc if startswith(n, "pop_mu_beta_pop.") && endswith(n, ".1"))

function cell_locations(adaptive, model)
    out = Dict{Any,Any}()
    for r in adaptive_centering_controls(adaptive, model)
        r.family === :ranef || continue
        key = (r.group, r.margin.coefficient)
        haskey(out, key) && @test out[key] == r.location   # one location per (group, term)
        out[key] = r.location
    end
    out
end

fd_gradient(f, q; h = 1e-6) = [begin
    p, m = copy(q), copy(q)
    p[i] += h; m[i] -= h
    (f(p) - f(m)) / (2h)
end for i in eachindex(q)]

function check_gradients(model, problem; trials = 3, kwargs...)
    base = adaptive_centering_problem(model, problem, AutoEnzyme(); kwargs...)
    n = length(WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(base)))
    d = LogDensityProblems.dimension(base)
    local_rng = Xoshiro(11)
    for _ in 1:trials
        controls = rand(local_rng, [0.0, 0.2, 0.5, 0.8, 1.0], n)
        wrapped = adaptive_centering_problem(model, problem, AutoEnzyme();
                                             centeredness = controls, kwargs...)
        q = 0.3 .* randn(local_rng, d)
        lp, grad = LogDensityProblems.logdensity_and_gradient(wrapped, q)
        @test lp ≈ LogDensityProblems.logdensity(wrapped, q)
        fd = fd_gradient(v -> LogDensityProblems.logdensity(wrapped, v), q)
        @test maximum(abs.(grad .- fd) ./ (1 .+ abs.(fd))) < 1e-5
    end
end

@testset "each cell is centred on its matching population coefficient" begin
    rp = adaptive_centering_problem(M_MAIN, P_MAIN, AutoEnzyme())
    locs = cell_locations(rp, M_MAIN)
    beta = BRM.brm_population_effect_coordinates(brm_descriptor(M_MAIN), :mu, UNC_MAIN;
                                                 coefficient = :x)
    @test locs[(:g, :Intercept)] == intercept_name(UNC_MAIN)
    @test locs[(:g, :x)] == UNC_MAIN[only(beta.coordinates)]
    @test locs[(:h, :Intercept)] == intercept_name(UNC_MAIN)
    @test locs[(:h, :t)] === nothing                     # no population `t`
end

@testset "nested cells take their enclosing total; centred-compiled and S2Z-absorbed cells 0" begin
    nested = SBBRMI(merge(@brm(begin
        mu ~ 1 + (1 | band) + (1 | g)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(data); mod = @__MODULE__, total_groups = ())
    pn = instantiate(nested)
    rows = [r for r in adaptive_centering_controls(
                adaptive_centering_problem(nested, pn, AutoEnzyme()), nested)
            if r.family === :ranef]
    icpt = intercept_name(BS.param_unc_names(pn.model))
    @test all(r -> r.location == icpt, (r for r in rows if r.group === :band))
    for r in rows
        r.group === :g || continue                       # `g` nests in `band`
        b = band[findfirst(==(string(r.level)), g)]
        @test r.location == icpt * " + " *
              only(x.name for x in rows if x.group === :band && string(x.level) == b)
    end

    centred = SBBRMI(M_MAIN.parent; mod = @__MODULE__, total_groups = (),
                     centered_groups = [:g])
    pc = instantiate(centred)
    locs = cell_locations(adaptive_centering_problem(centred, pc, AutoEnzyme()), centred)
    @test locs[(:g, :Intercept)] === nothing && locs[(:g, :x)] === nothing
    @test locs[(:h, :Intercept)] == intercept_name(BS.param_unc_names(pc.model))

    s2z = SBBRMI(merge(@brm(begin
        mu ~ 1 + (1 | band) + (1 | h)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(data); mod = @__MODULE__, total_groups = (), s2z_groups = [:band], s2z_rho = 1.0)
    ps = instantiate(s2z)
    locs = cell_locations(adaptive_centering_problem(s2z, ps, AutoEnzyme()), s2z)
    @test locs[(:h, :Intercept)] === nothing             # intercept absorbed by S2Z
end

@testset "the map in closed form" begin
    blocks = BRM.adaptive_centering_blocks(M_MAIN, UNC_MAIN)
    n = length(WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(
        adaptive_centering_problem(M_MAIN, P_MAIN, AutoEnzyme()))))
    local_rng = Xoshiro(5)
    for _ in 1:3
        controls = rand(local_rng, [0.0, 0.4, 1.0], n)
        rp = adaptive_centering_problem(M_MAIN, P_MAIN, AutoEnzyme(); centeredness = controls)
        rows = adaptive_centering_controls(rp, M_MAIN)
        q = 0.25 .* randn(local_rng, LogDensityProblems.dimension(rp))
        _, model = WarmupHMC.reparametrizer(rp)(q)
        values = BS.param_constrain(P_MAIN.model, model)
        names = BS.param_names(P_MAIN.model)
        value(name) = values[only(findall(==(name), names))]
        beta_of(r) = r.location === nothing ? 0.0 : q[coordinate(r.location)]
        for block in blocks
            K = block.ranef.n_terms
            binding = block.ranef.binding
            # `log_scale` / `tau.k` are `lower=0`: the unconstrained value is the log SD.
            tau = [exp(q[i]) for i in block.log_scales]
            L = K == 1 ? ones(1, 1) : [value("$(binding)_L.$i.$j") for i in 1:K, j in 1:K]
            C = Diagonal(tau) * L
            for gi in 1:block.ranef.n_groups
                z = zeros(K)
                for k in 1:K
                    r = only(r for r in rows if r.index == block.effects[k, gi])
                    m = sum(C[k, j] * z[j] for j in 1:k-1; init = 0.0)
                    z[k] = (q[block.effects[k, gi]] - r.c * (m + beta_of(r))) / C[k, k]^r.c
                    @test model[block.effects[k, gi]] ≈ z[k] atol = 1e-12
                end
            end
        end
    end
end

@testset "gradients: hand-written K <= 2 pullback" begin
    rp = adaptive_centering_problem(M_MAIN, P_MAIN, AutoEnzyme())
    ext = Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt)
    @test first(WarmupHMC.reparametrizer(rp).pairs).second.args[1] isa
          ext.BRMAdaptiveCenteringArgument{:location,1}
    check_gradients(M_MAIN, P_MAIN)
end

const M_WIDE = SBBRMI(merge(@brm(begin
    mu ~ 1 + x + t + (1 + x + t | g)
    y ~ Normal(mu, sigma)
end), @brm(begin
    sigma ~ Exponential(1)
end))(data); mod = @__MODULE__, total_groups = ())
const P_WIDE = instantiate(M_WIDE)

@testset "gradients: block-level pass (K = 3) and learned order" begin
    locs = cell_locations(adaptive_centering_problem(M_WIDE, P_WIDE, AutoEnzyme()), M_WIDE)
    @test all(!isnothing, values(locs))
    check_gradients(M_WIDE, P_WIDE)
    check_gradients(M_WIDE, P_WIDE; ranef_order = :learned)
end

@testset "gradients: joint derived-scale wrapper (Student-t)" begin
    tblock = SBBRMI(merge(@brm(begin
        mu ~ 1 + x + (1 + x | p | g)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
        ranef(:, p) ~ TDist(4)
    end))(data); mod = @__MODULE__, total_groups = ())
    pt = instantiate(tblock)
    locs = cell_locations(adaptive_centering_problem(tblock, pt, AutoEnzyme()), tblock)
    @test locs[(:g, :Intercept)] == intercept_name(BS.param_unc_names(pt.model))
    check_gradients(tblock, pt)
end

@testset "centeredness = :midpoint" begin
    rp = adaptive_centering_problem(M_MAIN, P_MAIN, AutoEnzyme(); centeredness = :midpoint)
    @test all(r -> r.c == 0.5, adaptive_centering_controls(rp, M_MAIN))
    hs = SBBRMI(merge(@brm(begin
        mu ~ 1 + hsgp(x; k = 4) + (1 | g)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(data); mod = @__MODULE__, total_groups = ())
    ph = instantiate(hs)
    rows = adaptive_centering_controls(
        adaptive_centering_problem(hs, ph, AutoEnzyme(); centeredness = :midpoint), hs)
    @test all(r -> r.c == (r.family === :ranef ? 0.5 : r.target), rows)
    @test any(r -> r.family === :hsgp, rows)
    # refused: a misspelt start must not fall through to the compiled start
    # silently (dev §1); the only symbol is `:midpoint`.
    @test_throws ArgumentError adaptive_centering_problem(M_MAIN, P_MAIN, AutoEnzyme();
                                                          centeredness = :middle)
end
