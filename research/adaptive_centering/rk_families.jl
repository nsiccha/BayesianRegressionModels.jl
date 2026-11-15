# Adaptive partial centring of RKBRMI random effects, family by family (todo
# 0nsdowt): for each block family the RKBRMI wrapper newly covers, the plain RK
# density against `adaptive_centering_problem(rk, ...)` on a synthetic panel
# whose group levels the data pin, so the level ridge `level_g = location +
# scale * z_g` is the stiff direction. The panel is
# `test/rk_adaptive_centering.jl`'s: `G` groups of `n` rows, each a mean with a
# known standard error of 0.3-0.6 around levels of 60-130, one slope on `t`. The
# 0.2.263 measurement of the wrapper, `rk_panel.jl`, draws another panel of that
# shape with slopes on `t`, `c1` and `c2`.
#
#     julia --project=test research/adaptive_centering/rk_families.jl [family...]
#
# Families: student_t, known, bounded, stratified, factor. Each runs 4 chains
# through WarmupHMC's quorum sampler (`min_completed = 3`), 400 draws, for each
# seed, plain and wrapped, and prints max R-hat, min bulk ESS over every
# coordinate (RK layout frame), divergences and wall time.
using BayesianRegressionModels, Random, Statistics, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems, WarmupHMC
using DifferentiationInterface: AutoEnzyme
using CategoricalArrays, MCMCDiagnosticTools
const BRM = BayesianRegressionModels
const AD = AutoEnzyme(; mode=Enzyme.Reverse)

function panel(; G=12, n=30, seed=1)
    rng = Xoshiro(seed)
    level = 60 .+ 70 .* rand(rng, G)
    slope = 0.5 .* randn(rng, G)
    g = repeat(1:G; inner=n)
    t = repeat(collect(range(-1.0, 1.0; length=n)), G)
    se = 0.3 .+ 0.3 .* rand(rng, G * n)
    y = level[g] .+ (1.0 .+ slope[g]) .* t .+ se .* randn(rng, G * n)
    labels = string.("g", lpad.(g, 2, '0'))
    (; y, t, se, g=categorical(labels),
       b=categorical(string.("s", cld.(g, 4))),
       f=categorical(string.("f", mod1.(g, 3))))
end

const FAMILIES = Dict(
    # Student-t intercepts: each level's factor is `sqrt(w[g]) * tau`.
    :student_t => df -> @brm(df, begin
        mu ~ 1 + t + (1 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        ranef(:, g) ~ TDist(4.0)
        y ~ Normal(mu, se)
    end),
    # A known intercept SD: the ridge is `level_g = beta + 30 * z_g`.
    :known => df -> @brm(df, begin
        mu ~ 1 + t + (1 + t || g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g, Intercept) = 30.0
        sd(mu, g, t) ~ Exponential(2.0)
        y ~ Normal(mu, se)
    end),
    # A bounded intercept SD (`:interval` on RK).
    :bounded => df -> @brm(df, begin
        mu ~ 1 + t + (1 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g) ~ Uniform(0.0, 100.0)
        y ~ Normal(mu, se)
    end),
    # Three strata of four groups, each with its own intercept SD.
    :stratified => df -> @brm(df, begin
        mu ~ 1 + t + (1 | gr(g, by=b))
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end),
    # A factor constant within each group: the ridge also runs through its level.
    :factor => df -> @brm(df, begin
        mu ~ 1 + t + factor(f) + (1 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end),
)

# A run whose chains miss the completion quorum (too many stopped as stuck) is a
# result, not an error: it is reported as such.
function fit(lpdf; seed, n_chains=4, n_draws=400)
    rngs = [Xoshiro(seed + i) for i in 1:n_chains]
    out = nothing
    t = @elapsed try
        out = WarmupHMC.completion_warmup_mcmc(rngs, lpdf;
            min_completed=div(3 * n_chains, 4), n_draws)
    catch e
        occursin("quorum unmet", sprint(showerror, e)) || rethrow()
        return (; rhat=NaN, ess=NaN, divergent=-1,
                omitted=first(split(sprint(showerror, e), '\n')), seconds=NaN)
    end
    draws = [r.posterior_position for r in out.results]
    # draws × chains × parameters
    A = permutedims(cat((permutedims(d) for d in draws)...; dims=3), (1, 3, 2))
    (; rhat=maximum(MCMCDiagnosticTools.rhat(A)),
       ess=minimum(MCMCDiagnosticTools.ess(A)),
       divergent=sum(r.n_divergent_samples for r in out.results),
       omitted=out.completion.omitted_chain_indices, seconds=t)
end

function main(families)
    df = panel()
    for family in families
        rk = RKBRMI(FAMILIES[family](df))
        problem = rk_logdensity_problem(rk; ad_backend=AD)
        blocks = adaptive_centering_blocks(rk)
        println("== $family: $(length(blocks)) block(s), ",
                sum(b -> b.ranef.n_terms * b.ranef.n_groups, blocks), " cells, ",
                LogDensityProblems.dimension(problem), " coordinates")
        for seed in (101, 202)
            plain = fit(problem; seed)
            adaptive = fit(adaptive_centering_problem(rk, problem, AD); seed)
            for (label, r) in (("plain", plain), ("adaptive", adaptive))
                println(rpad("  seed $seed $label", 24),
                    "max R-hat ", round(r.rhat; digits=3),
                    "  min bulk ESS ", round(r.ess; digits=1),
                    "  divergent ", r.divergent,
                    "  omitted ", r.omitted,
                    "  ", round(r.seconds; digits=1), " s")
            end
        end
        flush(stdout)
    end
end

main(isempty(ARGS) ? [:student_t, :known, :bounded, :stratified, :factor] :
     Symbol.(ARGS))
