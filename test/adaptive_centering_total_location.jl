# test/adaptive_centering_total_location.jl — adaptive centring of a cell on
# the total of every term its level sits in (snag brm-zone-8-adapt-1b407672,
# decision
# `BayesianRegressionModels:snag.brm-zone-8-adapt-1b407672/decisions/2026-10-28T13-34-42-695-1wxltaq`).
#
# A level of a grouping that sits inside a factor level and inside coarser
# groupings of the same predictor is centred on the sum of the factor level's
# coordinate, the population coefficient, and the effect of each containing
# grouping's cell, so its `c = 1` coordinate is the level's total. The oracles
# do not read the wrapper's containment logic:
#
# - which coordinates each cell is centred on, from how the data were built;
# - the map in closed form, with every containing effect rebuilt by hand from
#   its own cell;
# - at `c = 1` the source coordinate of a pair equals its row's linear predictor
#   as BridgeStan computes it from the mapped point;
# - the inverse map returns the source point, and the wrapper's gradient equals
#   central finite differences of its own log density, on the per-pair, joint
#   (with an HSGP) and learned-order wrappers;
# - a level inside a correlated (`K = 2`) grouping keeps the rule from before:
#   centred on 0.
#
# Run: julia --project=test test/adaptive_centering_total_location.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan

instantiate(model) = Base.invokelatest(StanBlocks.stan_instantiate, model.model)
problem(model, p; kwargs...) = adaptive_centering_problem(model, p, AutoEnzyme(); kwargs...)

# `h = 1e-4`: with an HSGP term the joint wrapper's log density carries enough
# rounding that central differences at `1e-6` drift to ~1e-4 relative error
# on a model with no containing cells as well (it grows as `h` shrinks, so it
# is rounding, not the gradient); at `1e-4` both models agree to ~2e-6.
fd_gradient(f, q; h = 1e-4) = [begin
    p, m = copy(q), copy(q)
    p[i] += h; m[i] -= h
    (f(p) - f(m)) / (2h)
end for i in eachindex(q)]

function check_gradients(model, p; trials = 3, kwargs...)
    base = problem(model, p; kwargs...)
    n = length(WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(base)))
    d = LogDensityProblems.dimension(base)
    local_rng = Xoshiro(23)
    for _ in 1:trials
        controls = rand(local_rng, [0.0, 0.3, 0.7, 1.0], n)
        wrapped = problem(model, p; centeredness = controls, kwargs...)
        q = 0.3 .* randn(local_rng, d)
        lp, grad = LogDensityProblems.logdensity_and_gradient(wrapped, q)
        @test lp ≈ LogDensityProblems.logdensity(wrapped, q)
        fd = fd_gradient(v -> LogDensityProblems.logdensity(wrapped, v), q)
        @test maximum(abs.(grad .- fd) ./ (1 .+ abs.(fd))) < 1e-5
        ir = WarmupHMC.reparametrizer(wrapped)
        _, mapped = ir(q)
        _, back = WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
        @test back ≈ q atol = 1e-11
    end
end

ranef_rows(rp, model) = [r for r in adaptive_centering_controls(rp, model) if r.family === :ranef]
level_row(rows, group, level) =
    only(r for r in rows if r.group === group && string(r.level) == string(level))

# ---- weights: 0 + factor(cov) + (1 | sec) + (1 | pair) ----------------------
# Every (cov, sec) combination is one pair, so a pair sits in one factor level
# and one section, while a section holds pairs of every covariate (crossed).
rng = Xoshiro(17)
const NP = 30
const W_DATA = (; pair = ["p$i" for i in 1:NP],
                  sec = ["s$(mod1(i, 6))" for i in 1:NP],
                  cov = ["c$(cld(i, 6))" for i in 1:NP],
                  xx = randn(rng, NP), y = randn(rng, NP))

const M_W = SBBRMI(merge(@brm(begin
    lw ~ 0 + factor(cov) + (1 | sec) + (1 | pair)
    y ~ Normal(lw, sigma)
end), @brm(begin
    sigma ~ Exponential(1)
end))(W_DATA); mod = @__MODULE__, total_groups = ())
const P_W = instantiate(M_W)
const UNC_W = BS.param_unc_names(P_W.model)

# the factor level's cell-mean coordinate, by level value
function factor_coordinates(model, unc, predictor, column)
    r = BRM.brm_population_effect_coordinates(brm_descriptor(model), predictor, unc;
                                              coefficient = column)
    r.coding === :cellmeans ? Dict(string(c.level) => c.coordinate for c in r.cells) :
        Dict(string(c.nonreference_level) => c.coordinate for c in r.contrasts)
end

block_of(model, unc, group) = only(b for b in BRM.adaptive_centering_blocks(model, unc)
                                   if b.ranef.group === group)

@testset "a pair is centred on its covariate level plus its section's effect" begin
    rp = problem(M_W, P_W)
    rows = ranef_rows(rp, M_W)
    cov = factor_coordinates(M_W, UNC_W, :lw, :cov)
    @test all(r -> r.location === nothing, (r for r in rows if r.group === :sec))
    for i in 1:NP
        r = level_row(rows, :pair, W_DATA.pair[i])
        sec = level_row(rows, :sec, W_DATA.sec[i])
        @test r.location == UNC_W[cov[W_DATA.cov[i]]] * " + " * sec.name
    end
    # sections are mapped before the pairs that read them
    @test maximum(r.cell for r in rows if r.group === :sec) <
          minimum(r.cell for r in rows if r.group === :pair)
end

@testset "the map in closed form, and c = 1 samples the pair's total" begin
    cov = factor_coordinates(M_W, UNC_W, :lw, :cov)
    sb, pb = block_of(M_W, UNC_W, :sec), block_of(M_W, UNC_W, :pair)
    n = length(WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(problem(M_W, P_W))))
    local_rng = Xoshiro(5)
    for controls in (ones(n), rand(local_rng, [0.0, 0.4, 1.0], n))
        rp = problem(M_W, P_W; centeredness = controls)
        rows = ranef_rows(rp, M_W)
        q = 0.4 .* randn(local_rng, LogDensityProblems.dimension(rp))
        _, model = WarmupHMC.reparametrizer(rp)(q)
        c(group, level) = level_row(rows, group, level).c
        tau_s, tau_p = exp(q[only(sb.log_scales)]), exp(q[only(pb.log_scales)])
        effect = Dict{String,Float64}()
        for (g, level) in enumerate(string.(sb.ranef.levels))
            i = sb.effects[1, g]
            z = q[i] / tau_s^c(:sec, level)
            @test model[i] ≈ z atol = 1e-12
            effect[level] = tau_s * z
        end
        lw = BS.param_constrain(P_W.model, model; include_tp = true)
        names = BS.param_names(P_W.model; include_tp = true)
        lw_of(row) = lw[only(findall(==("lw.$row"), names))]
        for (g, level) in enumerate(string.(pb.ranef.levels))
            row = only(findall(==(level), W_DATA.pair))
            i = pb.effects[1, g]
            L = q[cov[W_DATA.cov[row]]] + effect[W_DATA.sec[row]]
            @test model[i] ≈ (q[i] - c(:pair, level) * L) / tau_p^c(:pair, level) atol = 1e-12
            # at c = 1 the source coordinate is the row's linear predictor
            all(==(1.0), controls) && @test q[i] ≈ lw_of(row) atol = 1e-12
        end
    end
end

@testset "gradients and inverse: per-pair wrapper" begin
    check_gradients(M_W, P_W)
end

@testset "a container declared after its pairs is still mapped first" begin
    late = SBBRMI(merge(@brm(begin
        lw ~ 0 + factor(cov) + (1 | pair) + (1 | sec)
        y ~ Normal(lw, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(W_DATA); mod = @__MODULE__, total_groups = ())
    pl = instantiate(late)
    rows = ranef_rows(problem(late, pl), late)
    @test maximum(r.cell for r in rows if r.group === :sec) <
          minimum(r.cell for r in rows if r.group === :pair)
    @test all(r -> occursin(" + ", r.location), (r for r in rows if r.group === :pair))
    check_gradients(late, pl; trials = 1)
end

@testset "joint wrapper (with an HSGP term)" begin
    joint = SBBRMI(merge(@brm(begin
        lw ~ 0 + factor(cov) + hsgp(xx; k = 4) + (1 | sec) + (1 | pair)
        y ~ Normal(lw, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(W_DATA); mod = @__MODULE__, total_groups = ())
    pj = instantiate(joint)
    rp = problem(joint, pj)
    @test any(r -> r.family === :hsgp, adaptive_centering_controls(rp, joint))
    rows = ranef_rows(rp, joint)
    @test all(r -> occursin(" + ", r.location), (r for r in rows if r.group === :pair))
    check_gradients(joint, pj)
end

# ---- a nested chain inside a treatment-coded factor -------------------------
# unit ⊂ band ⊂ region, and each unit sits in one level of `cov`; bands hold
# units of every covariate. `h` is crossed with everything.
const NU = 18
const unit_index = repeat(1:NU; inner = 4)
const N_DATA = (; unit = ["u$j" for j in unit_index],
                  band = ["b$(cld(j, 3))" for j in unit_index],
                  region = ["r$(cld(cld(j, 3), 3))" for j in unit_index],
                  cov = ["k$(mod1(j, 3))" for j in unit_index],
                  h = repeat(["h1", "h2", "h3"]; outer = 24),
                  x = randn(rng, 72), y = randn(rng, 72))

const M_N = SBBRMI(merge(@brm(begin
    mu ~ 1 + factor(cov) + (1 | region) + (1 | band) + (1 | unit)
    y ~ Normal(mu, sigma)
end), @brm(begin
    sigma ~ Exponential(1)
end))(N_DATA); mod = @__MODULE__, total_groups = ())
const P_N = instantiate(M_N)
const UNC_N = BS.param_unc_names(P_N.model)

intercept_name(unc) = only(n for n in unc if startswith(n, "pop_mu_beta_pop.") && endswith(n, ".1"))

@testset "nested chain: each level is centred on everything above it" begin
    rp = problem(M_N, P_N)
    rows = ranef_rows(rp, M_N)
    icpt = intercept_name(UNC_N)
    contrast = factor_coordinates(M_N, UNC_N, :mu, :cov)    # reference k1 has none
    first_row(col, level) = findfirst(==(level), N_DATA[col])
    for r in rows
        i = first_row(r.group, string(r.level))
        parts = [icpt]
        r.group === :unit && haskey(contrast, N_DATA.cov[i]) &&
            push!(parts, UNC_N[contrast[N_DATA.cov[i]]])
        r.group in (:band, :unit) && push!(parts, level_row(rows, :region, N_DATA.region[i]).name)
        r.group === :unit && push!(parts, level_row(rows, :band, N_DATA.band[i]).name)
        @test r.location == join(parts, " + ")
    end
end

@testset "nested chain: the map in closed form" begin
    icpt = only(findall(==(intercept_name(UNC_N)), UNC_N))
    contrast = factor_coordinates(M_N, UNC_N, :mu, :cov)
    rb, bb, ub = (block_of(M_N, UNC_N, g) for g in (:region, :band, :unit))
    n = length(WarmupHMC.reparam_controls(WarmupHMC.reparametrizer(problem(M_N, P_N))))
    local_rng = Xoshiro(9)
    for controls in (ones(n), rand(local_rng, [0.0, 0.5, 1.0], n))
        rp = problem(M_N, P_N; centeredness = controls)
        rows = ranef_rows(rp, M_N)
        q = 0.4 .* randn(local_rng, LogDensityProblems.dimension(rp))
        _, model = WarmupHMC.reparametrizer(rp)(q)
        first_row(col, level) = findfirst(==(level), N_DATA[col])
        effects = Dict{Symbol,Dict{String,Float64}}()
        for (group, block, above) in ((:region, rb, ()), (:band, bb, (:region,)),
                                      (:unit, ub, (:region, :band)))
            tau = exp(q[only(block.log_scales)])
            effects[group] = Dict{String,Float64}()
            for (g, level) in enumerate(string.(block.ranef.levels))
                i = first_row(group, level)
                L = q[icpt] + sum((effects[a][N_DATA[a][i]] for a in above); init = 0.0)
                group === :unit && haskey(contrast, N_DATA.cov[i]) &&
                    (L += q[contrast[N_DATA.cov[i]]])
                c = level_row(rows, group, level).c
                z = (q[block.effects[1, g]] - c * L) / tau^c
                @test model[block.effects[1, g]] ≈ z atol = 1e-12
                effects[group][level] = tau * z
            end
        end
    end
    check_gradients(M_N, P_N)
end

@testset "learned order beside a nested chain" begin
    learned = SBBRMI(merge(@brm(begin
        mu ~ 1 + x + factor(cov) + (1 | region) + (1 | band) + (1 | unit) + (1 + x | h)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(N_DATA); mod = @__MODULE__, total_groups = ())
    pl = instantiate(learned)
    rp = problem(learned, pl; ranef_order = :learned)
    @test WarmupHMC.reparametrizer(rp) isa
          Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt).BRMOrderedCenteringReparametrization
    @test all(r -> count("+", r.location) == 3 || count("+", r.location) == 2,
              (r for r in ranef_rows(rp, learned) if r.group === :unit))
    check_gradients(learned, pl; ranef_order = :learned)
end

@testset "a level inside a correlated grouping keeps location 0" begin
    corr = SBBRMI(merge(@brm(begin
        mu ~ 1 + x + (1 + x | band) + (1 | unit)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(N_DATA); mod = @__MODULE__, total_groups = ())
    pc = instantiate(corr)
    rows = ranef_rows(problem(corr, pc), corr)
    @test all(r -> r.location === nothing, (r for r in rows if r.group === :unit))
    @test all(r -> r.location !== nothing, (r for r in rows if r.group === :band))
end
