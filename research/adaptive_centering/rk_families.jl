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
# Families: student_t, known, bounded, stratified, factor, mm, r2d2, car_bym2, cdar,
# hsgp_mcycle, hsgp_grouped, hsgp_cor, hsgp_hyper. Each runs 4 chains
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

# A data-rich areal panel: a 6 x 6 grid of areas (rook neighbours), 30 rows
# per area with noise sd 0.3 around a BYM2-like field, so each area's total is
# pinned by its data.
function car_panel(; seed=3, n=30)
    rng = Xoshiro(seed)
    side = 6
    id(i, j) = (i - 1) * side + j
    from, to = Int[], Int[]
    for i in 1:side, j in 1:side
        i < side && (push!(from, id(i, j)); push!(to, id(i + 1, j)))
        j < side && (push!(from, id(i, j)); push!(to, id(i, j + 1)))
    end
    smooth = [sin(i / 2) + cos(j / 3) for i in 1:side for j in 1:side]
    effect = 2.0 .* smooth .+ 0.5 .* randn(rng, side^2)
    area = repeat(1:side^2; inner=n)
    y = 10 .+ effect[area] .+ 0.3 .* randn(rng, length(area))
    labels = string.("a", lpad.(1:side^2, 2, '0'))
    (; y, area=labels[area], from=labels[from], to=labels[to])
end

# A data-rich grouped walk: 6 groups x 20 weekly steps, 15 rows per step with
# noise sd 0.2, so each step's walk value is pinned by its data.
function walk_panel(; seed=4, P=6, W=20, n=15)
    rng = Xoshiro(seed)
    level = zeros(P, W)
    for p in 1:P, w in 1:W
        level[p, w] = (w == 1 ? 0.0 : 0.8 * level[p, w - 1]) + 0.5 * randn(rng)
    end
    group = repeat(1:P; inner=W * n)
    step = repeat(repeat(1:W; inner=n), P)
    y = 5 .+ [level[g, w] for (g, w) in zip(group, step)] .+ 0.2 .* randn(rng, length(group))
    (; y, patch=string.("p", group), week=step, C=Matrix(1.0I, P, P))
end

# The motorcycle acceleration data (`mcycle.csv`, 133 rows), both columns
# standardized so the default HSGP priors are on the data's scale.
function mcycle_data()
    rows = [split(l, ',') for l in readlines(joinpath(@__DIR__, "mcycle.csv"))[2:end]]
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    (; times=(times .- mean(times)) ./ std(times), accel=(accel .- mean(accel)) ./ std(accel))
end

# A data-rich grouped smooth: 4 groups, each its own curve on one axis, 80 rows
# per group with noise sd 0.1, so each group's basis weights are pinned by its
# data.
function hsgp_panel(; seed=5, G=4, n=80)
    rng = Xoshiro(seed)
    phase = 2pi .* rand(rng, G)
    group = repeat(1:G; inner=n)
    x = repeat(collect(range(-2.0, 2.0; length=n)), G)
    y = sin.(1.5 .* x .+ phase[group]) .+ 0.1 .* randn(rng, length(x))
    (; y, x, g=string.("g", group))
end

# Group curves whose levels are related: `G` curves of `n` rows on `[-2, 2]`
# with noise sd 0.1, phases `0.3 g` apart (so the levels' weights are strongly
# correlated) and amplitudes `amp` (so a per-level SD is the stiff direction
# when they differ).
function hsgp_level_panel(; seed=6, G=4, n=80, amp=ones(G))
    rng = Xoshiro(seed)
    group = repeat(1:G; inner=n)
    x = repeat(collect(range(-2.0, 2.0; length=n)), G)
    y = amp[group] .* sin.(1.5 .* x .+ 0.3 .* group) .+ 0.1 .* randn(rng, length(x))
    (; y, x, g=string.("g", group))
end

const FAMILY_DATA = Dict(:car_bym2 => car_panel, :cdar => walk_panel,
    :hsgp_mcycle => mcycle_data, :hsgp_grouped => hsgp_panel,
    :hsgp_cor => hsgp_level_panel,
    :hsgp_hyper => () -> hsgp_level_panel(; amp=[0.3, 0.8, 1.5, 3.0]))

const FAMILIES = Dict(
    # One smooth over a noisy, data-rich axis: the HSGP basis weights' cells.
    :hsgp_mcycle => df -> @brm(df, begin
        mu ~ 1 + hsgp(times; k=20)
        sigma ~ Exponential(1.0)
        accel ~ Normal(mu, sigma)
    end),
    # One smooth per group, the weights of each level pinned by its rows.
    :hsgp_grouped => df -> @brm(df, begin
        mu ~ 1 + hsgp(x; k=10, by=g)
        y ~ Normal(mu, 0.1)
    end),
    # Correlated group curves: the level cells of a `cor = Omega` term.
    :hsgp_cor => df -> @brm(df, begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + hsgp(x; k=10, by=g, cor=Omega)
        y ~ Normal(mu, 0.1)
    end),
    # Group curves of different amplitudes: a per-level SD's level cells.
    :hsgp_hyper => df -> @brm(df, begin
        mu ~ 1 + hsgp(x; k=10, by=g)
        log(sd(hsgp(x))) ~ 1 + (1 | g)
        y ~ Normal(mu, 0.1)
    end),
    # BYM2 areal field on a data-rich grid: theta cells centred on area totals.
    :car_bym2 => df -> @brm(df, begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        y ~ Normal(mu, 0.3)
    end),
    # A grouped AR(1) walk whose steps the data pin.
    :cdar => df -> @brm(df, begin
        mu ~ 1 + cdar(week; by=patch, cor=C)
        y ~ Normal(mu, 0.2)
    end),
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
        return (; rhat=NaN, worst=0, ess=NaN, divergent=-1,
                omitted=first(split(sprint(showerror, e), '\n')), seconds=NaN)
    end
    draws = [r.posterior_position for r in out.results]
    # draws × chains × parameters
    A = permutedims(cat((permutedims(d) for d in draws)...; dims=3), (1, 3, 2))
    rhats = MCMCDiagnosticTools.rhat(A)
    (; rhat=maximum(rhats), worst=argmax(rhats),
       ess=minimum(MCMCDiagnosticTools.ess(A)),
       divergent=sum(r.n_divergent_samples for r in out.results),
       omitted=out.completion.omitted_chain_indices, seconds=t)
end

function main(families)
    for family in families
        df = get(FAMILY_DATA, family, panel)()
        rk = RKBRMI(FAMILIES[family](df))
        problem = rk_logdensity_problem(rk; ad_backend=AD)
        # Field families adapt their car / cdar cells only under `fields = true`,
        # grouped HSGP level cells only under `hsgp_levels = true`.
        kw = family in (:car_bym2, :cdar) ? (; fields=true) :
             family in (:hsgp_cor, :hsgp_hyper) ? (; hsgp_levels=true) : (;)
        cells = length(adaptive_centering_controls(
            adaptive_centering_problem(rk, problem, AD; kw...), rk))
        println("== $family: $cells cells, ",
                LogDensityProblems.dimension(problem), " coordinates")
        init = family in DATA_STARTS ? data_start(rk, df) : missing
        names = String.(BRM._rk_layout_coordinate_names(rk))
        seeds = parse.(Int, split(get(ENV, "RK_FAMILY_SEEDS", "101,202"), ','))
        for seed in seeds
            plain = fit(problem; seed, init)
            adaptive = fit(adaptive_centering_problem(rk, problem, AD; kw...); seed, init)
            for (label, r) in (("plain", plain), ("adaptive", adaptive))
                println(rpad("  seed $seed $label", 24),
                    "max R-hat ", round(r.rhat; digits=3),
                    "  min bulk ESS ", round(r.ess; digits=1),
                    "  divergent ", r.divergent,
                    "  omitted ", r.omitted,
                    "  worst ", r.worst == 0 ? "-" : names[r.worst],
                    "  ", round(r.seconds; digits=1), " s")
            end
        end
        flush(stdout)
    end
end

main(isempty(ARGS) ? [:student_t, :known, :bounded, :stratified, :factor, :mm, :r2d2] :
     Symbol.(ARGS))
