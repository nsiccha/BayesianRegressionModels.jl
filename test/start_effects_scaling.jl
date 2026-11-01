# brm_start's explicit random-effect solve on the Stan backend is linear in the
# model's size and in the given levels: it constrains one point at a time into
# one buffer, locates the block's carriers through one name index, and
# measures a level-correlated block through levels + margins - 1 probes
# (snag brm-start-explic-aaf34bf9: margins + 1 full vectors held twice over
# took 50 GB on a 623k-parameter model).
# RUN: julia --project=test test/start_effects_scaling.jl
using BayesianRegressionModels, BridgeStan, LinearAlgebra, Random, Test
import StanBlocks
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

const SCALING_CACHE = mktempdir()
compiled(sb, name) = StanBlocks.stan_instantiate(sb.model;
    path=joinpath(SCALING_CACHE, "$name.stan")).model

# A compiled model that counts the points it constrains. `brm_start` reaches
# BridgeStan only through these calls.
struct Counting
    model::BridgeStan.StanModel
    constrained::Base.RefValue{Int}
end
Counting(model) = Counting(model, Ref(0))
for f in (:param_unc_num, :param_num, :param_names, :param_unc_names, :param_unconstrain)
    @eval BridgeStan.$f(c::Counting, args...; kwargs...) = BridgeStan.$f(c.model, args...; kwargs...)
end
function BridgeStan.param_constrain(c::Counting, args...; kwargs...)
    c.constrained[] += 1
    BridgeStan.param_constrain(c.model, args...; kwargs...)
end
function BridgeStan.param_constrain!(c::Counting, args...; kwargs...)
    c.constrained[] += 1
    BridgeStan.param_constrain!(c.model, args...; kwargs...)
end
# Points constrained by `f(model)`.
constrained_points(f, model) = (c = Counting(model); f(c); c.constrained[])

# G subjects, each with a series of T observations read from the subject's
# own PK parameters: the transformed parameters (`pred`) hold G × T values,
# dwarfing the block, as a large model's do. `C` correlates the subjects.
function scaling_data(rng, G, T)
    (; subject=collect(1:G), dose=fill(100.0, G), weight=0.3 .* randn(rng, G),
       t=[collect(range(0.5, 24.0; length=T)) for _ in 1:G],
       dv=[0.5 .+ abs.(randn(rng, T)) for _ in 1:G],
       C=[0.5^abs(i - j) for i in 1:G, j in 1:G])
end

# One margin of `|pk|`; the volume is one shared scalar.
one_margin(data) = @brm data begin
    sigma ~ Exponential(1)
    log_V ~ Normal(log(30), 1)
    log_CL ~ 1 + weight + (1 | pk | subject)
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V)
        pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
        dv[i] ~ normal(pred[i], sigma)
    end
end

# Four margins of `|pk|`.
four_margins(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | pk | subject)
    log_V ~ 1 + (1 | pk | subject)
    log_ka ~ 1 + (1 | pk | subject)
    log_F ~ 1 + (1 | pk | subject)
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V[i])
        ka = exp(log_ka[i])
        F = exp(log_F[i])
        pred[i] = F * dose[i] / Vc .* exp.(-(CL / Vc) .* t[i]) .* (1 .- exp.(-ka .* t[i]))
        dv[i] ~ normal(pred[i], sigma)
    end
end

# Two margins of `|pk|`, the subjects correlated by `C`.
correlated_margins(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | pk | gr(subject; cor=C))
    log_V ~ 1 + (1 | pk | gr(subject; cor=C))
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V[i])
        pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
        dv[i] ~ normal(pred[i], sigma)
    end
end

# Effects of the first `r` subjects, one vector in `ranefcoefnames` order.
given_effects(rng, K, r) = [i => 0.3 .* randn(rng, K) for i in 1:r]

# The effects `brm_ranef_draws` reads at the Stan point `s`, by level.
function ranef_effects(sb, model, s; id)
    names = String.(BridgeStan.param_names(model; include_tp=true))
    draws = permutedims(BridgeStan.param_constrain(model, s; include_tp=true))
    p = only(brm_ranef_draws(brm_descriptor(sb), draws, names; id))
    Dict(l => p.effects[1, i, :] for (i, l) in enumerate(p.levels))
end

function with_tp(model)
    BridgeStan.param_num(model; include_tp=true)
end

# Extra bytes `brm_start` allocates for the effects over the same start
# without them: the least of three measurements each, after a warm-up call.
function extra_allocation(sb, model, inits, base)
    without = NamedTuple{filter(!=(:pk), keys(inits))}(inits)
    least(x) = (brm_start(sb, model, x; base);
        minimum(@allocated(brm_start(sb, model, x; base)) for _ in 1:3))
    least(inits) - least(without)
end

const G = 24
const T = 2000
const DATA = scaling_data(Xoshiro(1), G, T)

@stestset "an independent block constrains margins + 2 points, whatever the levels" begin
    for (builder, K, name) in ((one_margin, 1, "one_margin"), (four_margins, 4, "four_margins"))
        sb = SBBRMI(builder(DATA); mod=@__MODULE__, total_groups=())
        block = only(ranef_blocks(brm_descriptor(sb)))
        @test block.n_terms == K && block.n_groups == G
        model = compiled(sb, name)
        @test with_tp(model) >= G * T
        base = 0.3 .* randn(Xoshiro(2), BridgeStan.param_unc_num(model))
        baseline = constrained_points(c -> brm_start(sb, c, (;); base), model)
        for r in (2, G)
            effects = given_effects(Xoshiro(r), K, r)
            inits = (; pk=(; effects))
            n = constrained_points(c -> brm_start(sb, c, inits; base), model)
            @test n - baseline == K + 2
            s = brm_start(sb, model, inits; base)
            read = ranef_effects(sb, model, s; id=:pk)
            for (level, e) in effects
                @test read[level] ≈ e rtol=1e-10 atol=1e-12
            end
            # Unlisted levels keep their standardized draws at the base.
            z = ranef_coordinates(block, String.(BridgeStan.param_unc_names(model)))
            rest = vec(z[:, r+1:end])
            @test s[rest] == base[rest]
        end
    end
end

@stestset "a level-correlated block constrains levels + margins + 1 points" begin
    sb = SBBRMI(correlated_margins(DATA); mod=@__MODULE__, total_groups=())
    block = only(ranef_blocks(brm_descriptor(sb)))
    @test BRM._ranef_level_correlated(block.family)
    @test block.n_terms == 2 && block.n_groups == G
    model = compiled(sb, "correlated_margins")
    base = 0.3 .* randn(Xoshiro(3), BridgeStan.param_unc_num(model))
    baseline = constrained_points(c -> brm_start(sb, c, (;); base), model)
    for r in (1, 2, G)
        effects = given_effects(Xoshiro(10 + r), 2, r)
        inits = (; pk=(; effects))
        n = constrained_points(c -> brm_start(sb, c, inits; base), model)
        @test n - baseline == r + 2 + 1
        s = brm_start(sb, model, inits; base)
        read = ranef_effects(sb, model, s; id=:pk)
        for (level, e) in effects
            @test read[level] ≈ e rtol=1e-10 atol=1e-12
        end
        z = ranef_coordinates(block, String.(BridgeStan.param_unc_names(model)))
        rest = vec(z[:, r+1:end])
        @test s[rest] == base[rest]
    end
end

@stestset "adding margins costs less than one constrained-plus-transformed vector" begin
    extra = Dict{Int,Int}()
    sizes = Dict{Int,Int}()
    data = scaling_data(Xoshiro(1), G, 20_000)
    for (builder, K, name) in ((one_margin, 1, "one_margin"), (four_margins, 4, "four_margins"))
        sb = SBBRMI(builder(data); mod=@__MODULE__, total_groups=())
        model = compiled(sb, name)
        base = 0.3 .* randn(Xoshiro(4), BridgeStan.param_unc_num(model))
        inits = (; pk=(; effects=given_effects(Xoshiro(5), K, G)))
        extra[K] = extra_allocation(sb, model, inits, base)
        sizes[K] = with_tp(model)
    end
    @info "extra allocation for the effects" one_margin=extra[1] four_margins=extra[4] vector_bytes=8 * sizes[4]
    # Three more margins used to add 3 × 3 full vectors (the probes' constrained
    # points, their `hcat` and its `permutedims`): 9 × the bound below.
    @test extra[4] - extra[1] < 8 * sizes[4]
end

@stestset "a level-correlated block's cost grows linearly in the given levels" begin
    extra = Dict{Int,Int}()
    for G_ in (12, 48)
        data = scaling_data(Xoshiro(6), G_, 2000)
        sb = SBBRMI(correlated_margins(data); mod=@__MODULE__, total_groups=())
        model = compiled(sb, "correlated_margins")
        base = 0.3 .* randn(Xoshiro(7), BridgeStan.param_unc_num(model))
        inits = (; pk=(; effects=given_effects(Xoshiro(8), 2, G_)))
        extra[G_] = extra_allocation(sb, model, inits, base)
    end
    @info "extra allocation for all levels of a level-correlated block" levels_12=extra[12] levels_48=extra[48]
    # Four times the levels: linear growth is ~4×; the levels × margins probes
    # over a model that grows with the levels gave ~16×.
    @test extra[48] < 8 * extra[12]
end
