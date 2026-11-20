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

# A block's level-`g` factor `C = diag(tau) L` from its own sampled
# coordinates: `tau = exp(x[log_scales])` and Stan's free-value factor.
function ordinary_factor(x, b, g)
    K = b.ranef.n_terms
    L = K == 1 ? ones(eltype(x), 1, 1) : cpc_cholesky(x[b.cholesky_free], K)
    Diagonal(exp.(x[b.log_scales])) * L
end

# The independent source -> RK map of a set of blocks: per block (in the
# wrapper's pair order), per group, per term, `z_k = (u_k - c_k (m_k + loc_k)) /
# s_k^c_k` with `C = factor(x, block, g)` (`diag(tau) L` by default),
# `m_k = sum_{j<k} C[k, j] z_j`, `s_k = C[k, k]`, and `ljac = -sum c_k log s_k`.
# `loc(x, y, block, k, g)` is the cell's location, which may read the model
# coordinates `y` of blocks mapped before it.
function oracle_map(x, blocks, controls, loc; factor=ordinary_factor)
    y = copy(x)
    ljac = zero(eltype(x))
    p = 0
    for b in blocks
        K, G = b.ranef.n_terms, b.ranef.n_groups
        for g in 1:G
            C = factor(x, b, g)
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

function check_exact(rk, adaptive, blocks, controls, loc; rng=Xoshiro(7),
                     factor=ordinary_factor, x=nothing)
    problem = adaptive.problem
    names = layout_names(rk)
    n = length(names)
    if isnothing(x)
        x = 0.3 .* randn(rng, n)
        # Keep the population coefficients near the data so the density is finite.
        for b in blocks, k in 1:b.ranef.n_terms, g in 1:b.ranef.n_groups
            x[b.effects[k, g]] += 0.5 * randn(rng)
        end
    end
    @test isfinite(LogDensityProblems.logdensity(problem, x))
    ljac, y = WarmupHMC.reparametrizer(adaptive)(copy(x))
    oljac, oy = oracle_map(x, blocks, controls, loc; factor)
    @test y ≈ oy rtol=1e-12 atol=1e-12
    @test ljac ≈ oljac rtol=1e-12 atol=1e-12
    rewritten = Set(Iterators.flatten(vec(b.effects) for b in blocks))
    @test all(y[j] == x[j] for j in 1:n if !(j in rewritten))
    wrapped = x -> (r = oracle_map(x, blocks, controls, loc; factor);
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

@stestset "a cdar walk beside the block: passes through, or cells after the block's under fields" begin
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
    locations = BRM._adaptive_ranef_locations(rk, blocks, names)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc(x, y, b, k, g) = x[only(locations[b].coefficient)]
    # By default the walk passes through: every walk coordinate is outside the
    # rewritten set.
    controls = rand(Xoshiro(10), 12)
    adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
    @test all(r -> r.family === :ranef, adaptive_centering_controls(adaptive, rk))
    check_exact(rk, adaptive, blocks, controls, loc)
    # `fields = true`: the walk's innovations are cells after the block's.
    adaptive = adaptive_centering_problem(rk, problem, AD; fields=true)
    rows = adaptive_centering_controls(adaptive, rk)
    @test [r.family for r in rows] == vcat(fill(:ranef, 12), fill(:cdar, 60))
    @test all(r -> occursin(".eta.", r.name), rows[13:end])
    # With the walk's cells at the compiled frame (`c = 0`), the block's cells
    # map exactly as without the walk; the walk's own map is checked below.
    controls = vcat(rand(Xoshiro(10), 12), zeros(60))
    check_exact(rk, adaptive_centering_problem(rk, problem, AD; centeredness=controls,
                                               fields=true), blocks, controls, loc)
end

# The scalar cells of `car` fields and `cdar` walks (`fields = true`), against an independent map
# from RK's own constrained values: each cell `y = (x - c * loc) * s^(-c)`, the
# log-Jacobian `-c * log s`. `cells` holds `(effect, loc, log_s)` functions of
# the source point, in the wrapper's order.
function scalar_oracle(x, cells, controls)
    y, ljac = copy(x), zero(eltype(x))
    for ((effect, loc, log_s), c) in zip(cells, controls)
        y[effect] = (x[effect] - c * loc(x)) * exp(-c * log_s(x))
        ljac -= c * log_s(x)
    end
    ljac, y
end

function check_scalar_cells(rk, cells; seeds=(31, 32))
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    n, d = length(cells), length(layout_names(rk))
    for seed in seeds
        controls = rand(Xoshiro(seed), n)
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls,
                                              fields=true)
        @test length(adaptive_centering_controls(adaptive, rk)) == n
        x = 0.3 .* randn(Xoshiro(seed + 100), d)
        ljac, y = WarmupHMC.reparametrizer(adaptive)(copy(x))
        oljac, oy = scalar_oracle(x, cells, controls)
        @test y ≈ oy rtol=1e-12 atol=1e-12
        @test ljac ≈ oljac rtol=1e-12 atol=1e-12
        wrapped(v) = (r = scalar_oracle(v, cells, controls);
                      r[1] + LogDensityProblems.logdensity(problem, r[2]))
        value, gradient = LogDensityProblems.logdensity_and_gradient(adaptive, x)
        @test value ≈ wrapped(x) rtol=1e-10
        fd = map(1:d) do j
            p, m = copy(x), copy(x)
            p[j] += 1e-6; m[j] -= 1e-6
            (wrapped(p) - wrapped(m)) / 2e-6
        end
        @test gradient ≈ fd rtol=1e-5 atol=1e-5
    end
end

# Areas "a".."h": component {a, b, c, d, h}, component {e, f}, the island "g"
# (with rows) and "h" (no rows).
const CAR_EDGES = (["a", "b", "c", "a", "d", "e"], ["b", "c", "d", "c", "h", "f"])
car_data() = (; area=["a", "b", "c", "d", "e", "f", "g", "a", "e"],
    y=[0.3, -0.2, 0.5, 1.1, -0.7, 0.0, 0.4, -0.1, 0.9],
    from=CAR_EDGES[1], to=CAR_EDGES[2])

@stestset "car fields: BYM2 theta cells on the area total, ICAR phi cells" begin
    data = car_data()
    for type in (:bym2, :icar)
        brmi = type === :bym2 ? (@brm data begin
            mu ~ 1 + car(area; edges=(from, to), type=:bym2)
            y ~ Normal(mu, 1.0)
        end) : (@brm data begin
            mu ~ 1 + car(area; edges=(from, to), type=:icar)
            y ~ Normal(mu, 1.0)
        end)
        rk = RKBRMI(brmi)
        names = layout_names(rk)
        record = only(r for r in BRM._rk_coordinate_records(rk.plan) if r.kind === :car)
        term = only(t for p in rk.plan.predictors for t in p.terms if t.kind === :car)
        decl = String(record.declaration)
        S = term.options.n_struct
        struct_pos = rk.plan.columns[term.options.struct_pos]
        inv_scale = rk.plan.columns[term.options.inv_scale]
        field(x) = getproperty(BRM._rk_constrained_values(rk, x), record.declaration)
        intercept = position(names, "mu_Intercept")
        cells = if type === :bym2
            # Area k (sorted labels): theta, scale sigma * sqrt(1 - rho) (sigma for
            # an island), centred on the intercept plus its spatial component.
            map(eachindex(struct_pos)) do k
                j = struct_pos[k] <= S ? struct_pos[k] : 0
                effect = position(names, "$decl.theta.$k")
                loc(x) = x[intercept] + (j == 0 ? 0.0 :
                    field(x).sigma * sqrt(field(x).rho * inv_scale[k]) *
                    x[position(names, "$decl.phi.$j")])
                log_s(x) = log(field(x).sigma) + (j == 0 ? 0.0 : log1p(-field(x).rho) / 2)
                (effect, loc, log_s)
            end
        else
            [(position(names, "$decl.phi.$j"), x -> 0.0, x -> log(field(x).sigma))
             for j in 1:S]
        end
        @test length(cells) == (type === :bym2 ? 8 : 7)
        check_scalar_cells(rk, cells)
        # Without `fields`, a model with only a field has nothing to adapt.
        @test_throws r"no random-effect block" adaptive_centering_problem(rk,
            rk_logdensity_problem(rk; ad_backend=AD), AD)
    end
end

# A scaled ICAR field (`type=:icar, scaled=true`, snag brm-car-icar-has-d25d72ec):
# its `phi` cells scaled by `sigma / sqrt(s_c)`, centred on 0, then the island
# "g"'s `theta` cell with scale `sigma`, centred on the intercept.
@stestset "car fields: scaled ICAR phi cells by 1/sqrt(s_c), island theta on the intercept" begin
    data = car_data()
    rk = RKBRMI(@brm data begin
        mu ~ 1 + car(area; edges=(from, to), type=:icar, scaled=true)
        y ~ Normal(mu, 1.0)
    end)
    names = layout_names(rk)
    record = only(r for r in BRM._rk_coordinate_records(rk.plan) if r.kind === :car)
    term = only(t for p in rk.plan.predictors for t in p.terms if t.kind === :car)
    decl = String(record.declaration)
    S = term.options.n_struct
    struct_pos = rk.plan.columns[term.options.struct_pos]
    inv_scale = rk.plan.columns[term.options.inv_scale]
    area_of = [findfirst(==(j), struct_pos) for j in 1:S]
    field(x) = getproperty(BRM._rk_constrained_values(rk, x), record.declaration)
    intercept = position(names, "mu_Intercept")
    cells = vcat(
        [(position(names, "$decl.phi.$j"), x -> 0.0,
          x -> log(field(x).sigma) + log(inv_scale[area_of[j]]) / 2) for j in 1:S],
        [(position(names, "$decl.theta.1"), x -> x[intercept], x -> log(field(x).sigma))])
    @test length(cells) == 8
    check_scalar_cells(rk, cells)
    rows = adaptive_centering_controls(adaptive_centering_problem(rk,
        rk_logdensity_problem(rk; ad_backend=AD), AD; fields=true), rk)
    @test [r.term for r in rows] == vcat(fill(1, S), 2)
    @test last(rows).level == "g"
end

@stestset "a cdar walk alone: innovation scales are the walk's marginal spread" begin
    data = (; week=repeat(1:5, 4), patch=repeat(["p", "q", "r", "s"], inner=5),
              y=sin.(1:20) ./ 2, C=[1.0 0.3 0.0 0.0; 0.3 1.0 0.2 0.0;
                                    0.0 0.2 1.0 0.1; 0.0 0.0 0.1 1.0])
    rk = RKBRMI(@brm data begin
        mu ~ 1 + cdar(week; by=patch, cor=C)
        y ~ Normal(mu, 1.0)
    end)
    names = layout_names(rk)
    record = only(r for r in BRM._rk_coordinate_records(rk.plan) if r.kind === :cdar)
    decl = String(record.declaration)
    walk(x) = getproperty(BRM._rk_constrained_values(rk, x), record.declaration)
    Cdiag = diag(record.cor_factor * record.cor_factor')
    cells = [(position(names, "$decl.eta.$p.$w"), x -> 0.0,
              x -> (r = walk(x).rho; log(walk(x).sigma) + log(Cdiag[p]) / 2 +
                    log((1 - r^(2w)) / (1 - r^2)) / 2))
             for w in 1:5 for p in 1:4]
    check_scalar_cells(rk, cells)
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

# The level-`g` factor of a block read from RK's own constrained values of the
# block's draws declaration: `tau` (or `tau_<k>`, or a stratum's row), `L`
# (or a stratum's slice) and a Student-t level's weight `w[g]`. `known` holds a
# known scale's value by term, `known_cor` a known correlation matrix.
function constrained_factor(rk; known=Dict{Int,Float64}(), known_cor=nothing,
                            stratum=nothing)
    function factor(x, b, g)
        nt = getproperty(BRM._rk_constrained_values(rk, x), b.ranef.binding)
        K = b.ranef.n_terms
        s = isnothing(stratum) ? nothing : stratum(b)
        tau = map(1:K) do k
            haskey(known, k) && return known[k]
            hasproperty(nt, :tau) ? (isnothing(s) ? nt.tau[k] : nt.tau[s, k]) :
                (v = getproperty(nt, Symbol(:tau_, k)); isnothing(s) ? only(v) : v[s])
        end
        L = K == 1 ? ones(1, 1) :
            !isnothing(known_cor) ? Matrix(cholesky(Symmetric(known_cor)).L) :
            isnothing(s) ? nt.L : nt.L[:, :, s]
        w = hasproperty(nt, :w) ? sqrt(nt.w[g]) : 1.0
        w * Diagonal(tau) * L
    end
end

@stestset "Student-t blocks: each level's factor carries its mixing weight" begin
    df = panel(; seed=13)
    student = @brm df begin
        mu ~ 1 + t + (1 + t | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        ranef(:, g) ~ TDist(4.0)
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(student)
    names = layout_names(rk)
    b = only(adaptive_centering_blocks(rk))
    @test b isa R2D2AdaptiveCenteringBlock && b.ranef.n_terms == 2
    draws = String(b.ranef.binding)
    @test [names[j] for j in b.mixing] == ["$draws.w.$g" for g in 1:12]
    @test names[only(b.cholesky_free)] == "$draws.L.1"
    locations = BRM._adaptive_ranef_locations(rk, [b], names)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc(x, y, b, k, g) = x[locations[b].coefficient[k]]
    factor = constrained_factor(rk)
    for controls in (zeros(24), ones(24), rand(Xoshiro(14), 24))
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
        check_exact(rk, adaptive, [b], controls, loc; factor)
    end
    rows = adaptive_centering_controls(adaptive_centering_problem(rk, problem, AD), rk)
    @test length(rows) == 24 && all(r -> r.family === :ranef && r.c == 0.0, rows)
    # The learned order builds over the Student-t block too.
    learned = adaptive_centering_problem(rk, problem, AD; ranef_order=:learned)
    @test length(adaptive_centering_controls(learned, rk)) == 24
    # A scalar Student-t block.
    scalar = @brm df begin
        mu ~ 1 + t + (1 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        ranef(:, g) ~ TDist(4.0)
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(scalar)
    b = only(adaptive_centering_blocks(rk))
    @test length(b.mixing) == 12
    locations = BRM._adaptive_ranef_locations(rk, [b], layout_names(rk))
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc1(x, y, b, k, g) = x[locations[b].coefficient[k]]
    controls = rand(Xoshiro(15), 12)
    check_exact(rk, adaptive_centering_problem(rk, problem, AD; centeredness=controls),
                [b], controls, loc1; factor=constrained_factor(rk))
end

@stestset "known values: constant scales and a known correlation" begin
    df = panel(; seed=16)
    # A sampled intercept SD beside a known slope SD in one correlated block.
    partly = @brm df begin
        mu ~ 1 + t + (1 + t | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g, Intercept) ~ Exponential(30.0)
        sd(mu, g, t) = 0.5
        y ~ Normal(mu, se)
    end
    # A known correlation beside sampled SDs.
    correlation = @brm df begin
        mu ~ 1 + t + (1 + t | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        cor(:, g) = [1 0.3; 0.3 1]
        y ~ Normal(mu, se)
    end
    for (brmi, known, known_cor) in ((partly, Dict(2 => 0.5), nothing),
                                     (correlation, Dict{Int,Float64}(), [1 0.3; 0.3 1]))
        rk = RKBRMI(brmi)
        names = layout_names(rk)
        b = only(adaptive_centering_blocks(rk))
        @test b isa R2D2AdaptiveCenteringBlock && isempty(b.mixing)
        @test isempty(b.cholesky_free) == !isnothing(known_cor)
        isnothing(known_cor) ||
            @test b.cholesky_known ≈ cholesky(Symmetric(known_cor)).L
        for (k, v) in known
            @test b.scales[k].constant == log(v) && isempty(b.scales[k].linear)
        end
        locations = BRM._adaptive_ranef_locations(rk, [b], names)
        problem = rk_logdensity_problem(rk; ad_backend=AD)
        loc(x, y, b, k, g) = x[locations[b].coefficient[k]]
        factor = constrained_factor(rk; known, known_cor)
        for controls in (ones(24), rand(Xoshiro(17), 24))
            adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
            check_exact(rk, adaptive, [b], controls, loc; factor)
        end
    end
end

@stestset "bounded scales: the log of RK's constrained value" begin
    df = panel(; seed=18)
    bounded = @brm df begin
        mu ~ 1 + t + (1 + t || g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g, Intercept) ~ Uniform(0.0, 100.0)
        sd(mu, g, t) ~ truncated(Normal(0.0, 5.0); lower=0.1)
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(bounded)
    names = layout_names(rk)
    transforms = BRM._rk_layout_coordinate_transforms(rk)
    blocks = adaptive_centering_blocks(rk)
    @test length(blocks) == 2 && all(b -> b isa R2D2AdaptiveCenteringBlock, blocks)
    # Each scale is one bounded atom on its own coordinate, with RK's bounds.
    atoms = [only(only(b.scales).bounded) for b in blocks]
    @test sort([(a[3], a[4]) for a in atoms]) == [(0.0, 100.0), (0.1, Inf)]
    @test Set(transforms[a[2]] for a in atoms) == Set([:interval, :floored])
    locations = BRM._adaptive_ranef_locations(rk, blocks, names)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc(x, y, b, k, g) = x[locations[b].coefficient[k]]
    factor = constrained_factor(rk)
    for controls in (ones(24), rand(Xoshiro(19), 24))
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
        check_exact(rk, adaptive, blocks, controls, loc; factor)
    end
end

@stestset "stratified gr(g, by=b): one block per stratum" begin
    df = panel(; seed=20)
    stratum_of(v) = "s" * string(cld(parse(Int, String(v)[2:end]), 4))
    data = merge(df, (; b=categorical(stratum_of.(df.g))))
    stratified = @brm data begin
        mu ~ 1 + t + (1 + t | gr(g, by=b))
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g, Intercept) ~ Exponential(30.0)
        sd(mu, g, t) ~ Exponential(2.0)
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(stratified)
    names = layout_names(rk)
    blocks = adaptive_centering_blocks(rk)
    @test length(blocks) == 3
    @test all(b -> b isa AdaptiveCenteringBlock && b.ranef.n_groups == 4 &&
                   b.ranef.by === :b, blocks)
    @test [stratum_of.(b.ranef.levels) for b in blocks] ==
        [fill("s$s", 4) for s in 1:3]
    for (s, b) in enumerate(blocks)
        draws = String(b.ranef.binding)
        @test [names[j] for j in b.log_scales] == ["$draws.tau_$k.$s" for k in 1:2]
        @test names[only(b.cholesky_free)] == "$draws.L.1.$s"
    end
    locations = BRM._adaptive_ranef_locations(rk, blocks, names)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc(x, y, b, k, g) = x[locations[b].coefficient[k]]
    @test all(b -> all(!iszero, locations[b].coefficient), blocks)
    factor = constrained_factor(rk; stratum=b -> findfirst(x -> x === b, blocks))
    for controls in (ones(24), rand(Xoshiro(21), 24))
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
        check_exact(rk, adaptive, blocks, controls, loc; factor)
    end
end

@stestset "factor levels: an intercept cell is centred on its level's coefficient" begin
    df = panel(; seed=22)
    level_of(v) = "f" * string(mod1(parse(Int, String(v)[2:end]), 3))
    data = merge(df, (; f=categorical(level_of.(df.g))))
    model = @brm data begin
        mu ~ 1 + t + factor(f) + (1 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(model)
    names = layout_names(rk)
    b = only(adaptive_centering_blocks(rk))
    locations = BRM._adaptive_ranef_locations(rk, [b], names)
    intercept = only(locations[b].coefficient)
    record = only(r for r in BRM._rk_coordinate_records(rk.plan)
                  if r.kind === :population_block)
    coefficient = Dict(string(l) => position(names, "$(record.declaration).$i")
                       for (i, l) in enumerate(record.level_values))
    for (g, level) in enumerate(b.ranef.levels)
        expected = haskey(coefficient, level_of(level)) ?
            [coefficient[level_of(level)]] : Int[]
        @test locations[b].factors[1, g] == expected
    end
    @test any(!isempty, locations[b].factors)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    loc(x, y, b, k, g) = x[intercept] + sum(x[locations[b].factors[1, g]]; init=0.0)
    for controls in (ones(12), rand(Xoshiro(23), 12))
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
        check_exact(rk, adaptive, [b], controls, loc)
        rows = adaptive_centering_controls(adaptive, rk)
        @test all(r -> occursin(names[intercept], r.location), rows)
    end
end

@stestset "multi-membership blocks: one cell per joint level, centred on 0" begin
    df = panel(; seed=25)
    # A second membership column over the same levels, shifted by one group.
    # RK's multi-membership groupings read plain (not categorical) columns.
    shift(v) = "g" * lpad(mod1(parse(Int, String(v)[2:end]) + 1, 12), 2, '0')
    data = merge(df, (; g=String.(string.(df.g)), h=shift.(string.(df.g)),
                        x=0.5 .* df.t))
    for K in (1, 2)
        brmi = K == 1 ? (@brm data begin
                mu ~ 1 + t + (1 | mm(g, h))
                effect(mu, Intercept) ~ Normal(100, 50)
                y ~ Normal(mu, se)
            end) : (@brm data begin
                mu ~ 1 + t + x + (1 + x | mm(g, h))
                effect(mu, Intercept) ~ Normal(100, 50)
                y ~ Normal(mu, se)
            end)
        rk = RKBRMI(brmi)
        names = layout_names(rk)
        b = only(adaptive_centering_blocks(rk))
        @test b isa AdaptiveCenteringBlock && b.ranef.n_terms == K
        @test b.ranef.group == (:g, :h) && b.ranef.n_groups == 12
        draws = String(b.ranef.binding)
        @test [names[b.effects[k, 5]] for k in 1:K] == ["$draws.z.5.$k" for k in 1:K]
        # No record of an mm block reaches the other inventory readers.
        @test !any(r -> r.kind === :mm_ranef, BRM._rk_coordinate_records(rk.plan))
        locations = BRM._adaptive_ranef_locations(rk, [b], names)
        @test all(iszero, locations[b].coefficient)
        problem = rk_logdensity_problem(rk; ad_backend=AD)
        loc(x, y, b, k, g) = 0.0
        controls = rand(Xoshiro(26 + K), 12K)
        check_exact(rk, adaptive_centering_problem(rk, problem, AD; centeredness=controls),
                    [b], controls, loc; factor=constrained_factor(rk))
    end
end

# The level factor of an R2D2-scaled block from RK's constrained values of the
# budget it reads: margin `k`'s scale `reference * sqrt(phi[entry] * R2 / (1 -
# R2))`, or its own sampled scale, with the names its record gives.
function r2d2_factor(rk)
    record = only(r for r in BRM._rk_coordinate_records(rk.plan; adaptive=true)
                  if r.kind === :r2d2_ranef)
    function factor(x, b, g)
        cv = BRM._rk_constrained_values(rk, x)
        value(name) = getproperty(cv, name)
        tau = map(record.r2d2) do spec
            haskey(spec, :free) && return value(spec.free)
            ref = spec.reference isa Real ? spec.reference : value(spec.reference)
            r2 = value(spec.r2)
            ref * sqrt(value(spec.phi)[spec.entry] * r2 / (1 - r2))
        end
        K = b.ranef.n_terms
        L = K == 1 ? ones(1, 1) : getproperty(cv, b.ranef.binding).L
        Diagonal(collect(tau)) * L
    end
end

@stestset "R2D2-scaled blocks: the scale is derived from R2, the shares and the reference" begin
    df = panel(; seed=27)
    data = merge(df, (; x=0.5 .* df.t))
    sampled = @brm data begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + x + (1 + x | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1, reference_scale=sigma)
        y ~ Normal(mu, se)
    end
    constant = @brm data begin
        mu ~ 1 + x + (1 + x | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1, reference_scale=30.0)
        y ~ Normal(mu, se)
    end
    # A partial ICC statement: the slope margin samples its own scale.
    partial = @brm data begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + x + (1 + x | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g, Intercept) ~ r2d2(reference_scale=sigma)
        y ~ Normal(mu, se)
    end
    for (label, brmi) in (("sampled reference", sampled), ("constant reference", constant),
                          ("partial ICC", partial))
        rk = RKBRMI(brmi)
        names = layout_names(rk)
        b = only(adaptive_centering_blocks(rk))
        @test b isa R2D2AdaptiveCenteringBlock && b.ranef.n_terms == 2
        @test b.ranef.family in BRM._ADAPTIVE_R2D2_FAMILIES
        @test !any(r -> r.kind === :r2d2_ranef, BRM._rk_coordinate_records(rk.plan))
        # Every R2D2 margin reads one stick-breaking share.
        @test all(s -> all(a -> a.stick, s.simplex), b.scales)
        locations = BRM._adaptive_ranef_locations(rk, [b], names)
        problem = rk_logdensity_problem(rk; ad_backend=AD)
        loc(x, y, b, k, g) = locations[b].coefficient[k] == 0 ? 0.0 :
            x[locations[b].coefficient[k]]
        for controls in (ones(24), rand(Xoshiro(28), 24))
            adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
            check_exact(rk, adaptive, [b], controls, loc; factor=r2d2_factor(rk))
        end
        # `:midpoint` leaves R2D2 cells at the compiled frame, as on SBBRMI.
        rows = adaptive_centering_controls(
            adaptive_centering_problem(rk, problem, AD; centeredness=:midpoint), rk)
        @test all(r -> r.c == 0.0, rows)
    end
end

@stestset "blocks the RKBRMI wrapper does not build raise" begin
    df = panel()
    none = @brm df begin
        mu ~ 1 + t
        y ~ Normal(mu, se)
    end
    rk = RKBRMI(none)
    @test isempty(adaptive_centering_blocks(rk))
    @test_throws r"no random-effect block" adaptive_centering_problem(rk,
        rk_logdensity_problem(rk; ad_backend=AD), AD)
    # Correlated levels share one factor across levels: refused on both backends.
    data = (; t=collect(range(0.0, 10.0; length=36)), g=repeat(1:4, 9),
              y=randn(Xoshiro(24), 36))
    levels = @brm data begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + t + (0 + t | gr(g; cor=Omega))
        y ~ Normal(mu, 1.0)
    end
    @test_throws r"correlates the levels" adaptive_centering_blocks(RKBRMI(levels))
end
