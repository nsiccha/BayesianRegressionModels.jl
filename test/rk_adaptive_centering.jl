# Adaptive partial centring of RKBRMI random effects under WarmupHMC (snag
# rkbrmi-has-no-ce-3158a399): `adaptive_centering_problem(rk, problem, backend)`
# adapts every ordinary random-effect cell of `rk_logdensity_problem(rk)`.
# Checked here against an independent map of the plain RK log density: the
# blocks name the RK layout's draws, scales and factor coordinates, each cell is
# located on its population coefficient (and a nested grouping on its parent's
# effect), and the wrapped density, Jacobian and gradient are exact at random
# controls, with every other coordinate passed through. Blocks the RKBRMI
# wrapper does not build yet raise.
#
#     julia --project=test test/rk_adaptive_centering.jl [testset filter...]
using Test, BayesianRegressionModels, Distributions, Random, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems, WarmupHMC
using DifferentiationInterface: AutoEnzyme
using CategoricalArrays
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

const AD = AutoEnzyme(; mode=Enzyme.Reverse)

# A panel whose group levels the data pin: `G` groups of `n` rows, each row a
# mean with a known standard error of 0.3-0.6 around levels of 60-130.
function panel(; G=12, n=30, seed=1)
    rng = Xoshiro(seed)
    level = 60 .+ 70 .* rand(rng, G)
    slope = 0.5 .* randn(rng, G)
    g = repeat(1:G; inner=n)
    t = repeat(collect(range(-1.0, 1.0; length=n)), G)
    se = 0.3 .+ 0.3 .* rand(rng, G * n)
    y = level[g] .+ (1.0 .+ slope[g]) .* t .+ se .* randn(rng, G * n)
    (; y, t, se, g=categorical(string.("g", lpad.(g, 2, '0'))))
end

uncorrelated(df) = @brm df begin
    mu ~ 1 + t + (1 + t || g)
    effect(mu, Intercept) ~ Normal(100, 50)
    sd(:, g) ~ Exponential(30.0)
    y ~ Normal(mu, se)
end

correlated(df) = @brm df begin
    mu ~ 1 + t + (1 + t | g)
    effect(mu, Intercept) ~ Normal(100, 50)
    sd(:, g) ~ Exponential(30.0)
    y ~ Normal(mu, se)
end

function nested_panel(; seed=2)
    rng = Xoshiro(seed)
    band = repeat(1:4; inner=3 * 20)
    unit = repeat(1:12; inner=20)
    y = 50 .+ 10 .* randn(rng, 4)[band] .+ 3 .* randn(rng, 12)[unit] .+
        0.5 .* randn(rng, length(unit))
    (; y, band=categorical(string.("b", band)),
       unit=categorical(string.("u", lpad.(unit, 2, '0'))))
end

nested(df) = @brm df begin
    mu ~ 1 + (1 | band) + (1 | unit)
    effect(mu, Intercept) ~ Normal(50, 20)
    sigma ~ Exponential(1.0)
    y ~ Normal(mu, sigma)
end

layout_names(rk) = String.(BRM._rk_layout_coordinate_names(rk))
function position(names, name)
    j = findfirst(==(name), names)
    isnothing(j) && error("missing coordinate $name")
    j
end

# Stan's `cholesky_factor_corr[K]` from its free values (row-major CPCs).
function cpc_cholesky(w, K)
    L = zeros(eltype(w), K, K)
    m = 0
    for i in 1:K
        rest = one(eltype(w))
        for j in 1:i-1
            m += 1
            r = tanh(w[m])
            L[i, j] = r * sqrt(rest)
            rest -= L[i, j]^2
        end
        L[i, i] = sqrt(rest)
    end
    L
end

# The independent source -> RK map of a set of blocks: per block (in the
# wrapper's pair order), per group, per term, `z_k = (u_k - c_k (m_k + loc_k)) /
# s_k^c_k` with `C = diag(tau) L`, `m_k = sum_{j<k} C[k, j] z_j`, `s_k = C[k, k]`,
# and `ljac = -sum c_k log s_k`. `loc(x, y, block, k, g)` is the cell's location,
# which may read the model coordinates `y` of blocks mapped before it.
function oracle_map(x, blocks, controls, loc)
    y = copy(x)
    ljac = zero(eltype(x))
    p = 0
    for b in blocks
        K, G = b.ranef.n_terms, b.ranef.n_groups
        tau = exp.(x[b.log_scales])
        L = K == 1 ? ones(eltype(x), 1, 1) : cpc_cholesky(x[b.cholesky_free], K)
        C = Diagonal(tau) * L
        for g in 1:G
            z = zeros(eltype(x), K)
            for k in 1:K
                p += 1
                c = controls[p]
                m = sum((C[k, j] * z[j] for j in 1:k-1); init=zero(eltype(x)))
                s = C[k, k]
                z[k] = (x[b.effects[k, g]] - c * (m + loc(x, y, b, k, g))) / s^c
                y[b.effects[k, g]] = z[k]
                ljac -= c * log(s)
            end
        end
    end
    ljac, y
end

function check_exact(rk, adaptive, blocks, controls, loc; rng=Xoshiro(7))
    problem = adaptive.problem
    names = layout_names(rk)
    n = length(names)
    x = 0.3 .* randn(rng, n)
    # Keep the population coefficients near the data so the density is finite.
    for b in blocks, k in 1:b.ranef.n_terms, g in 1:b.ranef.n_groups
        x[b.effects[k, g]] += 0.5 * randn(rng)
    end
    ljac, y = WarmupHMC.reparametrizer(adaptive)(copy(x))
    oljac, oy = oracle_map(x, blocks, controls, loc)
    @test y ≈ oy rtol=1e-12 atol=1e-12
    @test ljac ≈ oljac rtol=1e-12 atol=1e-12
    rewritten = Set(Iterators.flatten(vec(b.effects) for b in blocks))
    @test all(y[j] == x[j] for j in 1:n if !(j in rewritten))
    wrapped = x -> (r = oracle_map(x, blocks, controls, loc);
                    r[1] + LogDensityProblems.logdensity(problem, r[2]))
    value, gradient = LogDensityProblems.logdensity_and_gradient(adaptive, x)
    @test value ≈ wrapped(x) rtol=1e-10
    step = 1e-6
    fd = map(1:n) do j
        plus, minus = copy(x), copy(x)
        plus[j] += step; minus[j] -= step
        (wrapped(plus) - wrapped(minus)) / (2step)
    end
    @test gradient ≈ fd rtol=1e-5 atol=1e-5
end

@stestset "uncorrelated || blocks: layout positions, locations, exact map" begin
    df = panel()
    rk = RKBRMI(uncorrelated(df))
    names = layout_names(rk)
    blocks = adaptive_centering_blocks(rk)
    @test length(blocks) == 2
    @test all(b -> b isa AdaptiveCenteringBlock && b.ranef.n_terms == 1 &&
                   b.ranef.n_groups == 12 && b.target_c == 0.0 &&
                   isempty(b.cholesky_free), blocks)
    @test all(b -> b.ranef.levels == levels(df.g), blocks)
    for b in blocks
        draws = String(b.ranef.binding)
        @test [names[b.effects[1, g]] for g in 1:12] == ["$draws.z.$g.1" for g in 1:12]
        @test endswith(names[only(b.log_scales)], ".tau.1")
    end
    margins = BRM._rk_adaptive_margins(rk)
    coefficients = [only(margins[b.ranef.binding]).coefficient for b in blocks]
    @test sort(coefficients) == [:Intercept, :t]
    locations = BRM._adaptive_ranef_locations(rk, blocks, names)
    population = Dict(:Intercept => "pop_mu.beta_pop.1", :t => "pop_mu.beta_pop.2")
    for (b, c) in zip(blocks, coefficients)
        @test names[only(locations[b].coefficient)] == population[c]
        @test isnothing(locations[b].containers)
    end
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc(x, y, b, k, g) = x[position(names, population[only(margins[b.ranef.binding]).coefficient])]
    rng = Xoshiro(3)
    for controls in (zeros(24), ones(24), rand(rng, 24))
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
        check_exact(rk, adaptive, blocks, controls, loc)
    end
    # `c = 1` is the level total: the source coordinate is `beta + tau * z`.
    adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=1.0)
    x = randn(Xoshiro(4), length(names))
    _, y = WarmupHMC.reparametrizer(adaptive)(copy(x))
    for b in blocks, g in 1:12
        beta = x[position(names, population[only(margins[b.ranef.binding]).coefficient])]
        @test x[b.effects[1, g]] ≈ beta + exp(x[only(b.log_scales)]) * y[b.effects[1, g]]
    end
    rows = adaptive_centering_controls(adaptive, rk)
    @test length(rows) == 24
    @test all(r -> r.family === :ranef && r.c == 1.0 && r.target == 0.0, rows)
    @test Set(r.location for r in rows) == Set(values(population))
    @test Set(r.margin.coefficient for r in rows) == Set([:Intercept, :t])
    @test [r.name for r in rows] == [names[r.index] for r in rows]
    # The default start is the emitted (non-centred) frame.
    @test all(r -> r.c == 0.0, adaptive_centering_controls(
        adaptive_centering_problem(rk, problem, AD), rk))
end

@stestset "correlated block: Cholesky coordinates and the triangular map" begin
    df = panel(; seed=5)
    rk = RKBRMI(correlated(df))
    names = layout_names(rk)
    blocks = adaptive_centering_blocks(rk)
    @test length(blocks) == 1
    b = only(blocks)
    @test b.ranef.n_terms == 2 && b.ranef.n_groups == 12
    @test length(b.cholesky_free) == 1 && length(b.log_scales) == 2
    draws = String(b.ranef.binding)
    @test names[only(b.cholesky_free)] == "$draws.L.1"
    @test [names[b.effects[k, 3]] for k in 1:2] == ["$draws.z.3.1", "$draws.z.3.2"]
    locations = BRM._adaptive_ranef_locations(rk, blocks, names)
    @test [names[j] for j in locations[b].coefficient] ==
        ["pop_mu.beta_pop.1", "pop_mu.beta_pop.2"]
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc(x, y, b, k, g) = x[locations[b].coefficient[k]]
    for controls in (ones(24), rand(Xoshiro(6), 24))
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
        check_exact(rk, adaptive, blocks, controls, loc)
    end
    learned = adaptive_centering_problem(rk, problem, AD; ranef_order=:learned)
    rows = adaptive_centering_controls(learned, rk)
    @test length(rows) == 24 && all(r -> r.rank in (1, 2), rows)
end

@stestset "nested grouping: a unit is centred on its band's effect" begin
    df = nested_panel()
    rk = RKBRMI(nested(df))
    names = layout_names(rk)
    blocks = adaptive_centering_blocks(rk)
    @test length(blocks) == 2
    locations = BRM._adaptive_ranef_locations(rk, blocks, names)
    band = only(b for b in blocks if b.ranef.group === :band)
    unit = only(b for b in blocks if b.ranef.group === :unit)
    @test isnothing(locations[band].containers)
    @test !isnothing(locations[unit].containers)
    for g in 1:12
        container, term, parent = only(locations[unit].containers[1, g])
        @test container === band && term == 1
        @test band.ranef.levels[parent] == "b$(cld(g, 3))"
    end
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    intercept = only(locations[band].coefficient)
    # Bands are mapped first; a unit is centred on the intercept plus its
    # band's effect `tau_band * z_band`, read from the band's mapped draw.
    ordered = [band, unit]
    loc(x, y, b, k, g) = b === band ? x[intercept] : begin
        _, _, parent = only(locations[unit].containers[1, g])
        x[intercept] + exp(x[only(band.log_scales)]) * y[band.effects[1, parent]]
    end
    for controls in (ones(16), rand(Xoshiro(8), 16))
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
        rows = adaptive_centering_controls(adaptive, rk)
        @test [r.group for r in rows] == vcat(fill(:band, 4), fill(:unit, 12))
        @test all(occursin(String(band.ranef.binding) * ".z.", r.location)
                  for r in rows if r.group === :unit)
        check_exact(rk, adaptive, ordered, controls, loc)
    end
end

@stestset "a cdar walk beside the block passes through" begin
    df = panel(; G=6, n=10, seed=9)
    data = merge(df, (; step=repeat(1:10, 6), I_g=Matrix(1.0I, 6, 6)))
    brmi = @brm data begin
        mu ~ 1 + t + (1 + t || g) + cdar(step; by = g, cor = I_g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(brmi)
    names = layout_names(rk)
    blocks = adaptive_centering_blocks(rk)
    @test length(blocks) == 2
    @test any(n -> occursin(".eta.", n), names)
    margins = BRM._rk_adaptive_margins(rk)
    locations = BRM._adaptive_ranef_locations(rk, blocks, names)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc(x, y, b, k, g) = x[only(locations[b].coefficient)]
    controls = rand(Xoshiro(10), 12)
    adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
    # Every walk coordinate is outside the rewritten set and passes through.
    check_exact(rk, adaptive, blocks, controls, loc)
end

# A block scale whose log lies below the normal range of `exp`, as Pathfinder's
# line searches reach from a random start: `exp(-750.0) == 0.0` and
# `exp(-720.0)` is subnormal, while the plain RK density can still be finite.
# At the compiled centring (`c = 0`, the default start) the wrapper is that
# density: map, value and gradient. Before BRM 0.2.268 a cell's log scale was
# `log(tau * stick) = log(0.0) = -Inf`, and `-Inf` times the zero exponent
# `t - c` made the map, and so the density, `NaN` there: the wrapped density's
# Pathfinder start was not the plain density's (snag rkbrmi-adaptive-326587b2).
# `value` is the log scale; the gradient is compared where the plain density's
# is finite, and that is every coordinate unless `finite_gradient = false`.
function check_compiled_frame(rk, underflow; value=-750.0, generic=true, off_frame=true,
                              finite_gradient=true)
    names = layout_names(rk)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    adaptive = adaptive_centering_problem(rk, problem, AD)
    x = 0.3 .* randn(Xoshiro(11), length(names))
    x[underflow] .= value
    @test exp(value) < floatmin(Float64)
    lp, g = LogDensityProblems.logdensity_and_gradient(problem, x)
    # The positive control: the model density is finite at this point.
    ok = isfinite.(g)
    @test isfinite(lp) && (!finite_gradient || all(ok))
    ljac, y = WarmupHMC.reparametrizer(adaptive)(copy(x))
    @test ljac == 0 && y == x
    lpw, gw = LogDensityProblems.logdensity_and_gradient(adaptive, x)
    @test lpw == lp
    @test gw[ok] == g[ok]
    if generic   # WarmupHMC's own differentiation of the map, without BRM's pass
        lpg, gg = invoke(LogDensityProblems.logdensity_and_gradient,
            Tuple{WarmupHMC.ReparametrizedProblem,AbstractVector}, adaptive, x)
        @test lpg == lp
        @test gg[ok] ≈ g[ok] rtol=1e-12 atol=1e-12
    end
    # Off the compiled frame the scale's log is its coordinate, so the cell map
    # stays finite: `c = 1/2` multiplies by `exp(-value / 2)`.
    off_frame || return
    half = adaptive_centering_problem(rk, problem, AD; centeredness=0.5)
    ljac, y = WarmupHMC.reparametrizer(half)(copy(x))
    @test isfinite(ljac) && all(isfinite, y)
end

@stestset "the compiled-frame start is the RK density where a scale underflows" begin
    df = panel(; G=6, n=10, seed=12)
    # `||`: scalar blocks, BRM's hand-written K <= 2 pullback.
    rk = RKBRMI(uncorrelated(df))
    blocks = adaptive_centering_blocks(rk)
    check_compiled_frame(rk, [only(b.log_scales) for b in blocks])
    # Correlated `K = 2`: both scales, through the same pullback. With the first
    # scale underflowed, term 2's innovation divides by `tau1^c`: `0^c` off the
    # compiled frame, and its derivative `0 * Inf` in WarmupHMC's generic pass, so
    # those two are checked with the second scale only.
    rk = RKBRMI(correlated(df))
    b = only(adaptive_centering_blocks(rk))
    check_compiled_frame(rk, b.log_scales; generic=false, off_frame=false)
    check_compiled_frame(rk, [b.log_scales[2]])
    # `K = 3`: the blockwise reverse pass, the last term's scale.
    data = merge(df, (; t2=df.t .^ 2))
    wide = @brm data begin
        mu ~ 1 + t + t2 + (1 + t + t2 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(wide)
    b = only(adaptive_centering_blocks(rk))
    @test b.ranef.n_terms == 3
    check_compiled_frame(rk, [b.log_scales[3]])
    # A unit nested in a band: the band's effect enters the unit's location. This
    # model's own density is not finite once the band's scale is `0.0`, and its
    # gradient in that scale not once the scale is subnormal, so the check is at
    # a subnormal scale and over the other coordinates of the gradient.
    rk = RKBRMI(nested(nested_panel()))
    band = only(b for b in adaptive_centering_blocks(rk) if b.ranef.group === :band)
    check_compiled_frame(rk, [only(band.log_scales)]; value=-720.0, finite_gradient=false)
end

@stestset "blocks the RKBRMI wrapper does not build raise" begin
    df = panel()
    student = @brm df begin
        mu ~ 1 + t + (1 | g)
        ranef(:, g) ~ TDist(4.0)
        y ~ Normal(mu, se)
    end
    @test_throws r"Student-t" adaptive_centering_blocks(RKBRMI(student))
    known = @brm df begin
        mu ~ 1 + t + (1 + t || g)
        sd(mu, g, t) = 0.5
        y ~ Normal(mu, se)
    end
    @test_throws r"known value" adaptive_centering_blocks(RKBRMI(known))
    none = @brm df begin
        mu ~ 1 + t
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(none)
    @test isempty(adaptive_centering_blocks(rk))
    @test_throws r"no ordinary random-effect block" adaptive_centering_problem(rk,
        rk_logdensity_problem(rk; ad_backend=AD), AD)
end
