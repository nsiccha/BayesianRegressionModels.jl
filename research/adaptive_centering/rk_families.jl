# Adaptive partial centring of RKBRMI random effects, family by family (todo
# 0nsdowt): for each block family the RKBRMI wrapper newly covers, the plain RK
# density against `adaptive_centering_problem(rk, ...)` on a synthetic panel
# whose group levels the data pin, so the level ridge `level_g = location +
# scale * z_g` is the stiff direction. The panel is the one
# `test/rk_adaptive_centering.jl` and the 0.2.263 measurement use: `G` groups of
# `n` rows, each a mean with a known standard error of 0.3-0.6 around levels of
# 60-130.
#
#     julia --project=test research/adaptive_centering/rk_families.jl [family...]
#
# Families: student_t, known, bounded, stratified, factor, mm, r2d2. Each runs 4 chains
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
    # `h`: a second membership column, the next group (RK's multi-membership
    # groupings read plain columns).
    (; y, t, se, g=categorical(labels),
       b=categorical(string.("s", cld.(g, 4))),
       f=categorical(string.("f", mod1.(g, 3))),
       gm=labels, h=string.("g", lpad.(mod1.(g .+ 1, G), 2, '0')))
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
    # Multi-membership intercepts: each row averages its own and the next
    # group's effect. As on SBBRMI, these cells stay centred on 0.
    :mm => df -> @brm(df, begin
        mu ~ 1 + t + (1 | mm(gm, h))
        effect(mu, Intercept) ~ Normal(100, 50)
        y ~ Normal(mu, se)
    end),
    # An R2D2-scaled intercept SD with a constant reference scale.
    :r2d2 => df -> @brm(df, begin
        mu ~ 1 + t + (1 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1, reference_scale=30.0)
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
# Families whose random starts WarmupHMC's initialization cannot leave: the
# R2D2 panel fails Pathfinder for the plain density and the wrapper alike (9
# retries per chain), so both start at the intercept's data mean, everything
# else 0 (in the compiled frame, which is the wrapper's default start too).
const DATA_STARTS = Set([:r2d2])
data_start(rk, df) = (x = zeros(length(BRM._rk_layout_coordinate_names(rk)));
    x[findfirst(==("pop_mu.beta_pop.1"), String.(BRM._rk_layout_coordinate_names(rk)))] =
        mean(df.y); x)

function fit(lpdf; seed, n_chains=4, n_draws=400, init=missing)
    rngs = [Xoshiro(seed + i) for i in 1:n_chains]
    out = nothing
    t = @elapsed try
        out = WarmupHMC.completion_warmup_mcmc(rngs, lpdf;
            min_completed=div(3 * n_chains, 4), n_draws,
            init=init === missing ? missing : [copy(init) for _ in 1:n_chains])
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
        init = family in DATA_STARTS ? data_start(rk, df) : missing
        for seed in (101, 202)
            plain = fit(problem; seed, init)
            adaptive = fit(adaptive_centering_problem(rk, problem, AD); seed, init)
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

main(isempty(ARGS) ? [:student_t, :known, :bounded, :stratified, :factor, :mm, :r2d2] :
     Symbol.(ARGS))
