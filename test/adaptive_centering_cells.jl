# The public description of adaptive centring's random-effect cells,
# `adaptive_centering_cells(model)` (snag public-rkbrmi-ad-4eeb6b28, decision
# `BayesianRegressionModels:snag.public-rkbrmi-ad-4eeb6b28/decisions/2026-11-15T08-00-53-597-0e4u3mv`).
#
# The parity fixture: `public_map` rebuilds the map of
# `adaptive_centering_problem(model, problem, ad_backend)` from the rows alone,
# reading nothing of BRM but those rows (no `BRM._` call, no block type), as a
# caller outside BRM must (ReactiveKernelMethods emits it for JAX). At random
# controls it reproduces the wrapper's map and log-Jacobian
# (`WarmupHMC.reparametrizer(adaptive)`), its log density, and that density's
# gradient (central finite differences of the public density against the
# wrapper's gradient). The rows are plain data and follow the wrapper's cell
# order, which `adaptive_centering_controls` reports.
#
# Covered: on RKBRMI, scalar `||` blocks, a correlated K = 2 block, K = 3, a unit
# nested in a band (containing cells), a cdar walk and car fields (0.2.316), and the
# derived-scale blocks of BRM 0.2.311 (Student-t, known scales and correlation,
# bounded scales, stratified blocks, factor-level locations, and a Student-t
# unit nested in an ordinary band through the joint wrapper), and those of
# 0.2.313 (multi-membership blocks; R2D2-scaled blocks, whose shares are
# stick-breaking simplex atoms); on
# SBBRMI, factor levels and crossed containing groupings, a Student-t block
# nested in an ordinary one beside a centred block with known SDs and
# correlation (the joint derived-scale wrapper), and an R2D2 block whose scales
# are unit and simplex atoms, and car cells beside a random-effect block and a
# cdar walk (0.2.316); HSGP basis-weight cells (squared-exponential, anisotropic,
# grouped, periodic, known hyperparameters) on both backends, beside a
# random-effect block; grouped HSGP level cells on both backends; SBBRMI
# least-squares-located exact totals and S2Z contrasts or group coordinates;
# projected locations and `rw` walk cells on both backends (0.2.387).
#
#     julia --project=test test/adaptive_centering_cells.jl [testset filter...]
include(joinpath(@__DIR__, "fitting_packages.jl"))
using Test, BayesianRegressionModels, Distributions, Random, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems, WarmupHMC
using DifferentiationInterface: AutoEnzyme
using CategoricalArrays
import StanBlocks
import SpecialFunctions
include(joinpath(@__DIR__, "testset_filter.jl"))

const AD = AutoEnzyme(; mode=Enzyme.Reverse)

# ---- the public map, from the rows alone ------------------------------------

logistic(v) = inv(1 + exp(-v))
softplus(v) = v > 0 ? v + log1p(exp(-v)) : log1p(exp(v))

# `log(simplex[entry])` of Stan's (>= 2.37) isometric log-ratio simplex transform
# of the `N` free values `y`: the sum-to-zero vector
# `z[j] = sum(y[i] / sqrt(i (i + 1)) for i in j:N) - (j - 1) y[j-1] / sqrt((j - 1) j)`,
# then the softmax.
function log_simplex(y, entry)
    N = length(y)
    z = [sum((y[i] / sqrt(i * (i + 1)) for i in j:N); init=zero(eltype(y))) -
         (j > 1 ? (j - 1) * y[j - 1] / sqrt((j - 1) * j) : zero(eltype(y)))
         for j in 1:N+1]
    m = maximum(z)
    z[entry] - (m + log(sum(exp.(z .- m))))
end

# `log(simplex[entry])` of ReactiveKernelsPPL's stick-breaking simplex of the
# `N` free values `u`: break fractions `z_i = σ(u_i + log(N + 1 - i))`, entry
# `j <= N` is `z_j * prod(1 - z_i for i < j)`, entry `N + 1` the remaining stick.
function log_stick(u, entry)
    N = length(u)
    z = [logistic(u[i] + log(N + 1 - i)) for i in 1:N]
    rest = sum(log1p(-z[i]) for i in 1:entry-1; init=zero(eltype(u)))
    entry <= N ? rest + log(z[entry]) : rest
end

function log_scale(spec, x)
    acc = spec.constant
    for a in spec.linear
        acc += a.weight * x[a.coordinate]
    end
    for a in spec.unit   # log σ(v) = -softplus(-v), log(1 - σ(v)) = -softplus(v)
        v = x[a.coordinate]
        acc -= a.log_weight * softplus(-v) + a.log1m_weight * softplus(v)
    end
    for a in spec.simplex
        acc += a.weight * (a.transform === :stick_breaking ?
            log_stick(x[a.coordinates], a.entry) : log_simplex(x[a.coordinates], a.entry))
    end
    for a in spec.bounded
        v = x[a.coordinate]
        acc += a.weight * (isfinite(a.upper) ?
            log(a.lower + (a.upper - a.lower) * logistic(v)) : log(exp(v) + a.lower))
    end
    acc
end

# Stan's `cholesky_factor_corr[K]` from its canonical partial correlations, row
# by row.
function corr_cholesky(cpc, K)
    L = zeros(eltype(cpc), K, K)
    m = 0
    for i in 1:K
        rest = one(eltype(cpc))
        for j in 1:i-1
            m += 1
            L[i, j] = cpc[m] * sqrt(rest)
            rest -= L[i, j]^2
        end
        L[i, i] = sqrt(rest)
    end
    L
end

function block_factor(b, x, g)
    K = b.n_terms
    tau = [exp(log_scale(b.log_scales[i], x)) for i in 1:K]
    cpc = isempty(b.correlation.cpc) ? tanh.(x[b.correlation.free]) : b.correlation.cpc
    L = K == 1 ? ones(eltype(x), 1, 1) : corr_cholesky(cpc, K)
    r = isempty(b.mixing) ? one(eltype(x)) : exp(x[b.mixing[g]] / 2)
    r .* (Diagonal(tau) * L)
end

# A hyperparameter's constrained value from its row `(; coordinate, lower, upper)`.
constrain(h, x) = isfinite(h.upper) ? h.lower + (h.upper - h.lower) * logistic(x[h.coordinate]) :
    h.lower + exp(x[h.coordinate])

# A `car` or `cdar` cell's location and log scale, from its row alone.
function scalar_cell(c, x)
    T = eltype(x)
    sigma = constrain(c.scale.sigma, x)
    if c.family === :car
        rho = c.scale.rho.coordinate == 0 ? zero(T) : constrain(c.scale.rho, x)
        log_s = log(sigma) + (c.scale.share ? log1p(-rho) / 2 : zero(T))
        loc = c.location.intercept == 0 ? zero(T) : x[c.location.intercept]
        c.location.phi == 0 ||
            (loc += sigma * sqrt(rho * exp(c.location.log_inv_scale)) * x[c.location.phi])
        return loc, log_s
    end
    rho = constrain(c.scale.rho, x)
    w = c.scale.step
    v = 1 - rho^2 <= 1e-12 ? T(w) : (1 - rho^(2w)) / (1 - rho^2)
    zero(T), log(sigma) + log(c.scale.cdiag) / 2 + log(v) / 2
end

# A hyperparameter's value from its row: a known one (`coordinate == 0`) is
# its `lower`.
hyper(h, x) = h.coordinate == 0 ? oftype(float(first(x)), h.lower) : constrain(h, x)

# An HSGP cell's log spectral scale, from its row alone, floored smoothly below
# -110 at -150.
function hsgp_log_scale(c, x)
    sc = c.scale
    sigma = hyper(sc.sigma, x)
    rho(a) = hyper(sc.length_scales[length(sc.length_scales) == 1 ? 1 : a], x)
    l = if sc.spectrum === :periodic
        log(sigma) + (log(2) + log(SpecialFunctions.besselix(sc.harmonic, 1 / rho(1)^2))) / 2
    else
        log(sigma) + sum(log(rho(a) * sqrt(2pi)) / 2 - rho(a)^2 * sc.omega2[a] / 4
                         for a in eachindex(sc.omega2))
    end
    l > -110 ? l : -150 + softplus(l + 150)
end

# Grouped level rows describe either a bounded hyperparameter or a predictor
# of its log value. The latter's level effect is already a named coordinate.
function level_hyper_log(h, x)
    h.kind === :bounded && return log(hyper(h, x))
    value = h.intercept == 0 ? zero(eltype(x)) : x[h.intercept]
    h.sd.coordinate == 0 ? value : value + hyper(h.sd, x) * x[h.z]
end

function level_factor_row(c, b, x)
    sc = c.scale
    l = level_hyper_log(sc.sigma, x)
    for a in eachindex(sc.omega2)
        h = sc.length_scales[length(sc.length_scales) == 1 ? 1 : a]
        rho = exp(level_hyper_log(h, x))
        h.kind === :log_predictor && (rho = max(rho, sc.rho_floor))
        l += log(rho * sqrt(2pi)) / 2 - rho^2 * sc.omega2[a] / 4
    end
    l = l > -110 ? l : -150 + softplus(l + 150)
    L = if b.correlation.kind === :none
        Matrix{eltype(x)}(I, b.n_levels, b.n_levels)
    else
        cpc = b.correlation.kind === :data ? b.correlation.cpc : tanh.(x[b.correlation.free])
        corr_cholesky(cpc, b.n_levels)
    end
    exp(l) .* L[c.level_index, :]
end

# Source `x` -> compiled `y` and the log-Jacobian, cell by cell in row order.
function public_map(x, plan, controls)
    T = eltype(x)
    y = copy(x)
    ljac = zero(T)
    s = zeros(T, length(plan.cells))
    loc = zeros(T, length(plan.cells))
    for c in plan.cells
        b = plan.blocks[c.block]
        if c.family === :scalar
            if b.kind === :total
                c.cell == first(b.cells) || continue
                indices = [plan.cells[p].effect for p in b.cells]
                ell = x[b.log_scale]
                a = exp.((1 .- controls[b.cells]) .* ell)
                W = isempty(b.design) ? zeros(T, length(indices), 0) : hcat(b.design...)
                A = Diagonal(a)
                locs = W * ((W' * A * W) \ (W' * A * x[indices]))
                y[indices] = locs + a .* (x[indices] - locs)
                ljac += sum(log, a) + logdet(W' * W) - logdet(W' * A * W)
            else
                t, cc, ell = c.target, controls[c.cell], x[c.scale.log_coordinate]
                y[c.effect] = x[c.effect] * exp((t - cc) * ell)
                ljac += (t - cc) * ell
            end
            continue
        end
        if c.family === :hsgp && b.kind === :levels
            C = level_factor_row(c, b, x)
            m = sum(C[l] * y[plan.cells[q].effect] for (l, q) in enumerate(c.conditioning);
                    init=zero(T))
            cc, sp = controls[c.cell], C[c.level_index]
            y[c.effect] = (x[c.effect] - cc * m) * sp^(-cc)
            ljac -= cc * log(sp)
            continue
        end
        if c.family === :hsgp   # centred on 0, at its own target
            log_s, t, cc = hsgp_log_scale(c, x), c.target, controls[c.cell]
            y[c.effect] = x[c.effect] * exp((t - cc) * log_s)
            ljac += (t - cc) * log_s
            s[c.cell] = exp(log_s)
            continue
        end
        if c.family !== :ranef && c.family !== :rw   # a scalar cell with target 0
            L, log_s = scalar_cell(c, x)
            cc = controls[c.cell]
            y[c.effect] = (x[c.effect] - cc * L) * exp(-cc * log_s)
            ljac -= cc * log_s
            s[c.cell], loc[c.cell] = exp(log_s), L
            continue
        end
        C = block_factor(b, x, c.level_index)
        t, k, cc = b.target, c.term, controls[c.cell]
        m = zero(T)
        for (l, q) in enumerate(c.conditioning)
            m += C[k, l] * (y[plan.cells[q].effect] - t * loc[q]) / s[q]^t
        end
        L = m + (c.location.coefficient == 0 ? zero(T) : x[c.location.coefficient])
        for t in c.location.terms
            L += t.log_scale == 0 ? t.weight * x[t.coordinate] :
                (t.weight * exp(x[t.log_scale])) * x[t.coordinate]
        end
        for a in c.location.containers
            ya = y[plan.cells[a].effect]
            L += plan.blocks[plan.cells[a].block].target == 0 ? s[a] * ya : ya
        end
        # walk cells are compiled non-centred: their effect is `s * y`
        for w in c.location.walks
            L += w.weight * (s[w.cell] * y[plan.cells[w.cell].effect])
        end
        sp = C[k, k]
        y[c.effect] = t * L + (x[c.effect] - cc * L) * sp^(t - cc)
        ljac += (t - cc) * log(sp)
        s[c.cell], loc[c.cell] = sp, L
    end
    ljac, y
end

# ---- checks -----------------------------------------------------------------

plain(::Union{Number,Symbol,AbstractString,Nothing}) = true
plain(x::Union{AbstractVector,Tuple}) = all(plain, x)
plain(x::NamedTuple) = all(plain, values(x))
plain(_) = false

# The cell's location as `adaptive_centering_controls` labels it.
weighted(w, name) = w == 1 ? name : string(round(w; sigdigits=6), "*", name)

function location_label(plan, c, names)
    parts = String[]
    c.location.coefficient == 0 || push!(parts, names[c.location.coefficient])
    for t in c.location.terms
        push!(parts, weighted(t.weight, t.log_scale == 0 ? names[t.coordinate] :
            "exp($(names[t.log_scale]))*" * names[t.coordinate]))
    end
    append!(parts, (plan.cells[a].name for a in c.location.containers))
    for w in c.location.walks
        push!(parts, weighted(w.weight, plan.cells[w.cell].name))
    end
    isempty(parts) ? nothing : join(parts, " + ")
end

function check_rows(plan, rows, names)
    @test plain(plan)
    @test [c.cell for c in plan.cells] == 1:length(plan.cells)
    @test [b.block for b in plan.blocks] == 1:length(plan.blocks)
    @test length(rows) == length(plan.cells)
    @test [r.family for r in rows] == [c.family for c in plan.cells]
    @test [r.index for r in rows] == [c.effect for c in plan.cells]
    @test [r.name for r in rows] == [c.name for c in plan.cells] ==
          [names[c.effect] for c in plan.cells]
    @test [r.binding for r in rows] == [c.binding for c in plan.cells]
    @test [r.term for r in rows] == [c.term for c in plan.cells]
    # HSGP and scalar cells carry their own compiled frames.
    @test [r.target for r in rows] ==
          [c.family in (:hsgp, :scalar) ? c.target : plan.blocks[c.block].target for c in plan.cells]
    @test all(plan.blocks[c.block].family === c.family for c in plan.cells)
    ranef = [r for r in rows if r.family === :ranef]
    rcells = [c for c in plan.cells if c.family === :ranef]
    @test [r.margin for r in ranef] == [c.margin for c in rcells]
    @test [r.location for r in ranef] == [location_label(plan, c, names) for c in rcells]
    @test [string(r.level) for r in rows if r.family === :car] ==
          [string(c.level) for c in plan.cells if c.family === :car]
    lcells = [c for c in plan.cells if c.family === :hsgp &&
              plan.blocks[c.block].kind === :levels]
    for c in lcells
        r = rows[c.cell]
        @test (r.group, r.level, r.rank, r.margin) ==
              (c.group, c.level, c.level_index, c.margin)
        @test length(c.conditioning) == c.level_index - 1
        @test all(q -> q < c.cell && plan.cells[q].term == c.term &&
                       plan.cells[q].block == c.block, c.conditioning)
    end
    for c in rcells
        @test all(<(c.cell), c.location.containers)
        @test all(<(c.cell), c.conditioning)
        @test length(c.conditioning) == c.term - 1
    end
end

# `fields` and `hsgp` are the RKBRMI wrapper's keywords (its car and cdar cells,
# its HSGP cells); `nothing` passes none, as for an SBBRMI.
function check_parity(model, problem, names; trials=2, rng=Xoshiro(7), fd=true,
                      fields=nothing, hsgp=nothing, hsgp_levels=nothing)
    kw = isnothing(fields) ? (;) : (; fields)
    isnothing(hsgp) || (kw = merge(kw, (; hsgp)))
    isnothing(hsgp_levels) || (kw = merge(kw, (; hsgp_levels)))
    plan = adaptive_centering_cells(model, names; kw...)
    n = length(plan.cells)
    check_rows(plan, adaptive_centering_controls(
        adaptive_centering_problem(model, problem, AD; unc_names=names, kw...), model;
        unc_names=names), names)
    d = length(names)
    for controls in (rand(rng, [0.0, 0.3, 0.7, 1.0], n),
                     [rand(rng, 0:0.1:1) for _ in 1:n])[1:trials]
        adaptive = adaptive_centering_problem(model, problem, AD; unc_names=names,
                                              centeredness=controls, kw...)
        x = 0.3 .* randn(rng, d)
        ljac, y = WarmupHMC.reparametrizer(adaptive)(copy(x))
        pljac, py = public_map(x, plan, controls)
        @test py ≈ y rtol=1e-12 atol=1e-12
        @test pljac ≈ ljac rtol=1e-12 atol=1e-12
        density(v) = (r = public_map(v, plan, controls);
                      r[1] + LogDensityProblems.logdensity(problem, r[2]))
        value, gradient = LogDensityProblems.logdensity_and_gradient(adaptive, x)
        @test value ≈ density(x) rtol=1e-10
        fd || continue
        h = 1e-6
        numeric = map(1:d) do j
            p, m = copy(x), copy(x)
            p[j] += h; m[j] -= h
            (density(p) - density(m)) / (2h)
        end
        # Central differences of a density of magnitude ~1e6 carry round-off
        # near 1e-5 per entry; the norm-wise tolerance is test/rk_adaptive_centering.jl's.
        @test gradient ≈ numeric rtol=1e-5 atol=1e-5
    end
    plan
end

# ---- RKBRMI -----------------------------------------------------------------

function panel(; G=12, n=30, seed=1)
    rng = Xoshiro(seed)
    level = 60 .+ 70 .* rand(rng, G)
    slope = 0.5 .* randn(rng, G)
    g = repeat(1:G; inner=n)
    t = repeat(collect(range(-1.0, 1.0; length=n)), G)
    se = 0.3 .+ 0.3 .* rand(rng, G * n)
    y = level[g] .+ (1.0 .+ slope[g]) .* t .+ se .* randn(rng, G * n)
    (; y, t, t2=t .^ 2, se, g=categorical(string.("g", lpad.(g, 2, '0'))))
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

function rk_parity(brmi; fields=false, hsgp=true, hsgp_levels=false)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    # The RK layout's coordinate names (brm-use, "RK readers").
    names = String.(ReactiveKernelsPPL.coordinate_names(rk.model.layout))
    # The default `unc_names` of the RKBRMI method is that layout.
    default = adaptive_centering_cells(rk; fields, hsgp, hsgp_levels)
    plan = check_parity(rk, problem, names; fields, hsgp, hsgp_levels)
    @test default == plan
    rk, plan, names
end

@stestset "RKBRMI: scalar || blocks" begin
    df = panel()
    rk, plan, names = rk_parity(@brm df begin
        mu ~ 1 + t + (1 + t || g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end)
    @test length(plan.blocks) == 2 && length(plan.cells) == 24
    @test all(b -> b.target == 0.0 && b.n_terms == 1 && isempty(b.mixing) &&
                   isempty(b.correlation.free) && b.levels == levels(df.g), plan.blocks)
    @test Set(names[c.location.coefficient] for c in plan.cells) ==
          Set(["pop_mu.beta_pop.1", "pop_mu.beta_pop.2"])
    @test all(c -> isempty(c.location.containers) && isempty(c.location.terms),
              plan.cells)
    for b in plan.blocks
        atom = only(only(b.log_scales).linear)
        @test atom.weight == 1.0 && endswith(names[atom.coordinate], ".tau.1")
    end
end

@stestset "RKBRMI: correlated K = 2 and K = 3" begin
    df = panel(; seed=5)
    rk, plan, names = rk_parity(@brm df begin
        mu ~ 1 + t + (1 + t | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end)
    b = only(plan.blocks)
    @test b.n_terms == 2 && length(b.correlation.free) == 1 && isempty(b.correlation.cpc)
    @test names[only(b.correlation.free)] == "$(b.binding).L.1"
    @test [c.term for c in plan.cells[1:2]] == [1, 2]
    @test plan.cells[2].conditioning == [1]
    _, wide, _ = rk_parity(@brm df begin
        mu ~ 1 + t + t2 + (1 + t + t2 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end)
    @test only(wide.blocks).n_terms == 3 && length(only(wide.blocks).correlation.free) == 3
    @test wide.cells[3].conditioning == [1, 2]
end

@stestset "RKBRMI: a unit nested in a band reads its band's cell" begin
    df = nested_panel()
    rk, plan, names = rk_parity(@brm df begin
        mu ~ 1 + (1 | band) + (1 | unit)
        effect(mu, Intercept) ~ Normal(50, 20)
        sigma ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end)
    @test [b.group for b in plan.blocks] == [:band, :unit]   # containers first
    band = [c for c in plan.cells if c.group === :band]
    for c in plan.cells
        c.group === :unit || continue
        a = only(c.location.containers)
        @test plan.cells[a].group === :band
        @test plan.cells[a].level == "b$(cld(parse(Int, String(c.level)[2:end]), 3))"
        @test c.location.coefficient == only(unique(x.location.coefficient for x in band))
    end
end

@stestset "RKBRMI: a cdar walk beside the block (fields: its innovations)" begin
    df = panel(; G=6, n=10, seed=9)
    C = [0.5^abs(i - j) for i in 1:6, j in 1:6]
    data = merge(df, (; step=repeat(1:10, 6), C_g=C))
    brmi = @brm data begin
        mu ~ 1 + t + (1 + t || g) + cdar(step; by = g, cor = C_g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end
    # By default the walk passes through.
    _, plan, _ = rk_parity(brmi)
    @test length(plan.cells) == 12 && all(c -> c.family === :ranef, plan.cells)
    rk, plan, names = rk_parity(brmi; fields=true)
    ranef = [c for c in plan.cells if c.family === :ranef]
    walk = [c for c in plan.cells if c.family === :cdar]
    @test length(ranef) == 12 && length(walk) == 60
    @test all(c -> occursin(".eta.", c.name), walk)
    @test maximum(c.cell for c in ranef) < minimum(c.cell for c in walk)
    @test [c.scale.step for c in walk[1:7]] == [1, 1, 1, 1, 1, 1, 2]
    @test all(c -> c.scale.cdiag ≈ 1.0, walk)
    @test only(b for b in plan.blocks if b.family === :cdar).n_groups == 6
end

# Areas "a".."h": component {a, b, c, d, h} (a–b, b–c, c–d, a–c, d–h),
# component {e, f}, the island "g" (with rows) and "h" (no rows).
const CAR_EDGES = (["a", "b", "c", "a", "d", "e"], ["b", "c", "d", "c", "h", "f"])
car_data() = (; area=["a", "b", "c", "d", "e", "f", "g", "a", "e"],
    y=[0.3, -0.2, 0.5, 1.1, -0.7, 0.0, 0.4, -0.1, 0.9],
    from=CAR_EDGES[1], to=CAR_EDGES[2])

@stestset "RKBRMI: car fields (fields), one cell per area" begin
    data = car_data()
    rk, plan, names = rk_parity(@brm(data, begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        y ~ Normal(mu, 1.0)
    end); fields=true)
    cells = [c for c in plan.cells if c.family === :car]
    @test length(cells) == 8 && all(c -> c.term == 2, cells)
    @test [c.level for c in cells] == ["a", "b", "c", "d", "e", "f", "g", "h"]
    # The island "g" has no spatial component and no `rho` share in its scale.
    g = only(c for c in cells if c.level == "g")
    @test g.location.phi == 0 && !g.scale.share
    @test all(c -> c.location.intercept != 0, cells)
    _, plan, _ = rk_parity(@brm(data, begin
        mu ~ 1 + car(area; edges=(from, to), type=:icar)
        y ~ Normal(mu, 1.0)
    end); fields=true)
    cells = [c for c in plan.cells if c.family === :car]
    @test length(cells) == 7 && all(c -> c.term == 1 && c.location.intercept == 0, cells)
end

@stestset "RKBRMI: Student-t blocks carry their levels' mixing coordinates" begin
    df = panel(; seed=13)
    _, plan, names = rk_parity(@brm df begin
        mu ~ 1 + t + (1 + t | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        ranef(:, g) ~ TDist(4.0)
        y ~ Normal(mu, se)
    end)
    b = only(plan.blocks)
    @test [names[j] for j in b.mixing] == ["$(b.binding).w.$g" for g in 1:12]
    @test names[only(b.correlation.free)] == "$(b.binding).L.1"
end

@stestset "RKBRMI: known scales and a known correlation" begin
    df = panel(; seed=16)
    _, partly, _ = rk_parity(@brm df begin
        mu ~ 1 + t + (1 + t | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g, Intercept) ~ Exponential(30.0)
        sd(mu, g, t) = 0.5
        y ~ Normal(mu, se)
    end)
    s = only(partly.blocks).log_scales
    @test s[2].constant == log(0.5) && isempty(s[2].linear) && length(s[1].linear) == 1
    _, known, _ = rk_parity(@brm df begin
        mu ~ 1 + t + (1 + t | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        cor(:, g) = [1 0.3; 0.3 1]
        y ~ Normal(mu, se)
    end)
    c = only(known.blocks).correlation
    @test isempty(c.free) && only(c.cpc) ≈ 0.3
end

@stestset "RKBRMI: bounded scales are bounded atoms" begin
    df = panel(; seed=18)
    _, plan, _ = rk_parity(@brm df begin
        mu ~ 1 + t + (1 + t || g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g, Intercept) ~ Uniform(0.0, 100.0)
        sd(mu, g, t) ~ truncated(Normal(0.0, 5.0); lower=0.1)
        y ~ Normal(mu, se)
    end)
    atoms = [only(only(b.log_scales).bounded) for b in plan.blocks]
    @test sort([(a.lower, a.upper) for a in atoms]) == [(0.0, 100.0), (0.1, Inf)]
end

@stestset "RKBRMI: a stratified block is one block per stratum" begin
    df = panel(; seed=20)
    stratum_of(v) = "s" * string(cld(parse(Int, String(v)[2:end]), 4))
    data = merge(df, (; b=categorical(stratum_of.(df.g))))
    _, plan, _ = rk_parity(@brm data begin
        mu ~ 1 + t + (1 + t | gr(g, by=b))
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(mu, g, Intercept) ~ Exponential(30.0)
        sd(mu, g, t) ~ Exponential(2.0)
        y ~ Normal(mu, se)
    end)
    @test length(plan.blocks) == 3 && all(b -> b.by === :b && b.n_levels == 4, plan.blocks)
end

@stestset "RKBRMI: an intercept cell reads its factor level's coefficient" begin
    df = panel(; seed=22)
    level_of(v) = "f" * string(mod1(parse(Int, String(v)[2:end]), 3))
    data = merge(df, (; f=categorical(level_of.(df.g))))
    _, plan, _ = rk_parity(@brm data begin
        mu ~ 1 + t + factor(f) + (1 | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end)
    @test any(c -> !isempty(c.location.terms), plan.cells)
    @test all(c -> length(c.location.terms) <= 1 && c.location.coefficient != 0,
              plan.cells)
    # a factor level holding the whole level enters with weight 1 and no scale
    @test all(c -> all(t -> t.weight == 1.0 && t.log_scale == 0, c.location.terms),
              plan.cells)
    # the row shape before 0.2.387 is gone: a reader of it fails on the field
    @test !hasproperty(first(plan.cells).location, :factors)
end

@stestset "RKBRMI: a Student-t unit nested in an ordinary band (joint wrapper)" begin
    df = nested_panel()
    _, plan, _ = rk_parity(@brm df begin
        mu ~ 1 + (1 | band) + (1 | unit)
        effect(mu, Intercept) ~ Normal(50, 20)
        ranef(:, unit) ~ TDist(4.0)
        sigma ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end)
    @test [b.group for b in plan.blocks] == [:band, :unit]   # ordinary blocks first
    @test isempty(plan.blocks[1].mixing) && length(plan.blocks[2].mixing) == 12
    @test all(c -> c.group === :band ||
                   plan.cells[only(c.location.containers)].group === :band, plan.cells)
end

# ---- SBBRMI -----------------------------------------------------------------

const BS = StanBlocks.BridgeStan
instantiate(model) = Base.invokelatest(StanBlocks.stan_instantiate, model.model)

function sb_parity(model; kwargs...)
    problem = instantiate(model)
    names = BS.param_unc_names(problem.model)
    check_parity(model, problem, names; kwargs...), names
end

@stestset "RKBRMI: multi-membership blocks, centred on 0" begin
    df = panel(; seed=25)
    shift(v) = "g" * lpad(mod1(parse(Int, String(v)[2:end]) + 1, 12), 2, '0')
    data = merge(df, (; g=string.(df.g), h=shift.(string.(df.g)), x=0.5 .* df.t))
    _, plan, _ = rk_parity(@brm data begin
        mu ~ 1 + t + x + (1 + x | mm(g, h))
        effect(mu, Intercept) ~ Normal(100, 50)
        y ~ Normal(mu, se)
    end)
    b = only(plan.blocks)
    @test b.group == (:g, :h) && b.n_terms == 2 && length(b.correlation.free) == 1
    @test all(c -> c.location.coefficient == 0 && isempty(c.location.containers),
              plan.cells)
end

@stestset "RKBRMI: R2D2-scaled blocks read stick-breaking shares" begin
    df = panel(; seed=27)
    data = merge(df, (; x=0.5 .* df.t))
    _, plan, names = rk_parity(@brm data begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + x + (1 + x | g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1, reference_scale=sigma)
        y ~ Normal(mu, se)
    end)
    b = only(plan.blocks)
    for s in b.log_scales
        @test names[only(s.linear).coordinate] == "sigma"
        @test only(s.unit).log_weight == 0.5 && only(s.unit).log1m_weight == -0.5
        @test only(s.simplex).transform === :stick_breaking
    end
end

function hsgp_data(; seed=31, n=60)
    rng = Xoshiro(seed)
    x = sort(randn(rng, n))
    x2 = randn(rng, n)
    t = 48 .* rand(rng, n)
    g = categorical(rand(rng, ["a", "b", "c"], n))
    u = categorical(rand(rng, string.("u", 1:5), n))
    (; x, x2, t, g, u, y=sin.(2x) .+ 0.3 .* x2 .+ 0.2 .* randn(rng, n))
end

@stestset "RKBRMI: HSGP basis-weight cells, every spectrum and grouping" begin
    data = hsgp_data()
    for (brmi, n_cells, spectrum) in (
            (@brm(data, begin
                mu ~ 1 + hsgp(x, x2; k=(3, 2), c=(1.5, 2.0), iso=false)
                y ~ Normal(mu, 0.2)
            end), 6, :exp_quad),
            (@brm(data, begin mu ~ 1 + hsgp(x; k=3, by=g); y ~ Normal(mu, 0.2) end),
             9, :exp_quad),
            (@brm(data, begin
                mu ~ 1 + hsgp(t; k=2, cov=:periodic, period=24.0)
                y ~ Normal(mu, 0.2)
            end), 4, :periodic),
            (@brm(data, begin
                mu ~ 1 + hsgp(x; k=4)
                length_scale(mu, hsgp(x)) = 0.8
                y ~ Normal(mu, 0.2)
            end), 4, :exp_quad))
        _, plan, _ = rk_parity(brmi)
        @test length(plan.cells) == n_cells
        @test all(c -> c.family === :hsgp && c.scale.spectrum === spectrum, plan.cells)
        @test all(c -> c.target == 0.0, plan.cells)
    end
    # A grouped term is one block per level, in the grouping's level order.
    _, plan, _ = rk_parity(@brm data begin
        mu ~ 1 + hsgp(x; k=3, by=g)
        y ~ Normal(mu, 0.2)
    end)
    @test [b.level_index for b in plan.blocks] == 1:3
    # A known length scale is coordinate 0 at both bounds.
    _, plan, _ = rk_parity(@brm data begin
        mu ~ 1 + hsgp(x; k=4)
        length_scale(mu, hsgp(x)) = 0.8
        y ~ Normal(mu, 0.2)
    end)
    @test all(c -> only(c.scale.length_scales) == (; coordinate=0, lower=0.8, upper=0.8),
              plan.cells)
end

@stestset "RKBRMI: HSGP cells after a random-effect block's; hsgp = false drops them" begin
    data = hsgp_data(; seed=33)
    brmi = @brm data begin
        mu ~ 1 + hsgp(x; k=4) + (1 | u)
        sigma ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end
    _, plan, _ = rk_parity(brmi)
    @test [c.family for c in plan.cells] == vcat(fill(:ranef, 5), fill(:hsgp, 4))
    _, plan, _ = rk_parity(brmi; hsgp=false)
    @test all(c -> c.family === :ranef, plan.cells) && length(plan.cells) == 5
    # A term mapped across its levels has cells only under hsgp_levels=true.
    rk = RKBRMI(@brm data begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(x; k=3, by=g, cor=Omega) + (1 | u)
        y ~ Normal(mu, 0.3)
    end)
    @test count(c -> c.family === :hsgp, adaptive_centering_cells(rk; hsgp_levels=true).cells) == 9
    @test all(c -> c.family === :ranef, adaptive_centering_cells(rk).cells)
    @test all(c -> c.family === :ranef, adaptive_centering_cells(rk; hsgp=false).cells)
end

# Projected locations and `rw` walk cells (snag adaptive-centrin-647119ba). A
# trip-like grouping with an intercept and a slope sits in a factor level,
# carries a covariate constant within it (`near`) and one varying around it
# (`lprev`), and lies in one step of an `rw(year)` walk. Each backend passes the
# public-map parity above, and the two describe the same map: the walk's
# innovations first, as `:rw` cells, and for every trip cell the same weights on
# its factor level, its covariates' within-trip fit and the walk cells it reads.
@stestset "RKBRMI and SBBRMI: projected covariates and rw walk cells" begin
    rng = Xoshiro(11)
    NT, NR = 24, 5
    near = repeat(Float64.(rand(rng, 0:1, NT)); inner=NR)
    data = (; trip=repeat(["t$(lpad(i, 2, '0'))" for i in 1:NT]; inner=NR),
              year=repeat([2021 + mod(i, 3) for i in 1:NT]; inner=NR),
              dtype=repeat([isodd(i) ? "a" : "b" for i in 1:NT]; inner=NR),
              near, pos=repeat(collect(1.0:NR), NT),
              lprev=near .+ 0.1 .* randn(rng, NT * NR), y=randn(rng, NT * NR))
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 0 + factor(dtype) + near + lprev + (1 + pos || trip) + rw(year)
        y ~ Normal(mu, sigma)
    end
    _, rk_plan, rk_names = rk_parity(brmi)
    sb_plan, _ = sb_parity(SBBRMI(brmi; mod=@__MODULE__))
    # What a trip cell reads, without the backend's coordinates: its walk cells
    # by their place in the walk, and its weights, in the order both backends
    # list them (factor level, `near`, `lprev`).
    function reads(plan)
        walk = [c.cell for c in plan.cells if c.family === :rw]
        Dict((string(c.level), c.margin.coefficient) =>
                 (; weights=[t.weight for t in c.location.terms],
                    scales=[t.log_scale for t in c.location.terms],
                    walks=[findfirst(==(w.cell), walk) for w in c.location.walks],
                    walk_weights=[w.weight for w in c.location.walks],
                    matched=c.location.coefficient != 0)
             for c in plan.cells if c.family === :ranef)
    end
    for plan in (rk_plan, sb_plan)
        @test plan.blocks[1].family === :rw
        @test [c.family for c in plan.cells[1:3]] == [:rw, :rw, :ranef]
        @test count(c -> c.family === :rw, plan.cells) == 2      # 3 years, 2 innovations
        @test length(plan.cells) == 2 + 2 * NT
        @test all(c -> c.location.coefficient == 0 && isempty(c.location.terms) &&
                       isempty(c.location.containers) && isempty(c.location.walks),
                  (c for c in plan.cells if c.family === :rw))
    end
    @test [rk_names[c.effect] for c in rk_plan.cells[1:2]] ==
          ["rw_mu_year.z.1", "rw_mu_year.z.2"]
    rk_reads, sb_reads = reads(rk_plan), reads(sb_plan)
    @test keys(rk_reads) == keys(sb_reads) && length(rk_reads) == 2 * NT
    for (key, sb) in sb_reads
        rk = rk_reads[key]
        @test rk.weights ≈ sb.weights rtol=1e-12
        @test all(iszero, rk.scales) && all(iszero, sb.scales)
        @test rk.walks == sb.walks && rk.walk_weights == sb.walk_weights
        @test !rk.matched && !sb.matched    # no population intercept, no `pos` coefficient
    end
    # The oracle the weights answer to: a trip's intercept and slope cells read
    # the least-squares fit of each covariate on its own rows' `(1, pos)`, the
    # constant `near` whole (or not at all), and the walk innovations up to its
    # year with weight 1.
    for lev in unique(data.trip)
        I = findall(==(lev), data.trip)
        fit = hcat(ones(NR), data.pos[I]) \ data.lprev[I]
        n0 = data.near[first(I)]
        intercept, slope = sb_reads[(lev, :Intercept)], sb_reads[(lev, :pos)]
        @test intercept.weights ≈ (n0 == 0 ? [1.0, fit[1]] : [1.0, 1.0, fit[1]]) rtol=1e-10
        @test slope.weights ≈ [fit[2]] rtol=1e-10
        steps = data.year[first(I)] - 2021
        @test intercept.walks == collect(1:steps) && intercept.walk_weights == ones(steps)
        @test isempty(slope.walks)
    end
    # A model whose only cells are a walk's innovations builds on both backends.
    walk_only = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + near + rw(year)
        y ~ Normal(mu, sigma)
    end
    _, rk_walk, _ = rk_parity(walk_only)
    sb_walk, _ = sb_parity(SBBRMI(walk_only; mod=@__MODULE__))
    @test [c.family for c in rk_walk.cells] == [c.family for c in sb_walk.cells] ==
          [:rw, :rw]
    # Beside S2Z contrasts the walk's cells still come first.
    s2z_plan, _ = sb_parity(SBBRMI((@brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + near + (1 | trip) + rw(year)
        y ~ Normal(mu, sigma)
    end); mod=@__MODULE__, total_groups=(), s2z_groups=[:trip], s2z_rho=0.0))
    @test [c.family for c in s2z_plan.cells] == vcat([:rw, :rw], fill(:scalar, NT - 1))
end

@stestset "SBBRMI: factor levels and crossed containing groupings" begin
    rng = Xoshiro(17)
    NP = 30
    data = (; pair = ["p$i" for i in 1:NP], sec = ["s$(mod1(i, 6))" for i in 1:NP],
              cov = ["c$(cld(i, 6))" for i in 1:NP], y = randn(rng, NP))
    model = SBBRMI(merge(@brm(begin
        lw ~ 0 + factor(cov) + (1 | sec) + (1 | pair)
        y ~ Normal(lw, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(data); mod=@__MODULE__, total_groups=())
    plan, names = sb_parity(model)
    pairs = [c for c in plan.cells if c.group === :pair]
    @test length(pairs) == NP
    @test all(c -> length(c.location.terms) == 1 && c.location.coefficient == 0, pairs)
    @test all(c -> length(c.location.containers) == 1 &&
                   plan.cells[only(c.location.containers)].group === :sec, pairs)
end

@stestset "SBBRMI: Student-t, known SDs and correlation, a centred block" begin
    rng = Xoshiro(7)
    G = 12
    g = repeat(["g$j" for j in 1:G]; inner=6)
    band = ["b$(cld(parse(Int, s[2:end]), 4))" for s in g]
    h = repeat(["h$j" for j in 1:3]; outer=24)
    data = (; g, band, h, x=randn(rng, 72), y=randn(rng, 72))
    model = SBBRMI(merge(@brm(begin
        mu ~ 1 + x + (1 | band) + (1 + x | p | g) + (1 + x | q | h)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
        ranef(:, p) ~ TDist(4)
        sd(:, q) = 0.4
        cor(:, q) = [1 0.3; 0.3 1]
    end))(data); mod=@__MODULE__, total_groups=(), centered_groups=[:h])
    plan, names = sb_parity(model)
    byg = Dict(b.group => b for b in plan.blocks)
    @test byg[:band].block == 1                       # ordinary blocks first
    @test length(byg[:g].mixing) == G && byg[:g].target == 0.0
    @test byg[:h].target == 1.0 && length(byg[:h].correlation.cpc) == 1 &&
          isempty(byg[:h].correlation.free)
    @test all(s -> isempty(s.linear) && isempty(s.unit), byg[:h].log_scales)
    @test all(c -> c.location.coefficient == 0 && isempty(c.location.terms) &&
                   isempty(c.location.containers) && isempty(c.location.walks),
              (c for c in plan.cells if c.group === :h))
    @test all(c -> c.term != 1 || plan.cells[only(c.location.containers)].group === :band,
              (c for c in plan.cells if c.group === :g))
end

@stestset "SBBRMI: an R2D2 block's unit and simplex scale atoms" begin
    data = (; subject=repeat([1, 2, 3, 4], inner=2), x=collect(range(-1.0, 1.0, length=8)),
              y=sin.(1:8) ./ 3)
    model = SBBRMI(@brm(begin
        sigma ~ Exponential(1)
        log_Vc ~ 1 + (1 | p | subject)
        log_k10 ~ 1 + (1 | p | subject)
        sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1, reference_scale=sigma)
        cor(:, p) ~ LKJCholesky(2, 2)
        y ~ Normal(log_Vc + log_k10, sigma)
    end)(data); mod=@__MODULE__, total_groups=())
    plan, names = sb_parity(model)
    b = only(plan.blocks)
    @test all(s -> !isempty(s.unit) && !isempty(s.simplex), b.log_scales)
    @test all(s -> all(a -> a.transform === :stan, s.simplex), b.log_scales)
end

@stestset "SBBRMI: car cells beside a random-effect block, and a cdar walk" begin
    data = merge(car_data(), (; grp=["u", "v", "u", "v", "u", "v", "u", "v", "u"],
                                x=[0.1, -0.4, 0.8, 0.3, -1.0, 0.5, 0.2, -0.3, 0.6]))
    model = SBBRMI(@brm(data, begin
        mu ~ 1 + x + (1 + x | grp) + car(area; edges=(from, to), type=:bym2)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__, total_groups=())
    plan, names = sb_parity(model)
    @test count(c -> c.family === :ranef, plan.cells) == 4
    @test count(c -> c.family === :car, plan.cells) == 8
    walk = (; week=[1, 2, 1, 2, 1, 2, 1, 2, 1], patch=["p", "p", "q", "q", "p", "q", "p", "q", "p"],
              y=data.y, C=[1.0 0.3; 0.3 1.0])
    model = SBBRMI(@brm(walk, begin
        mu ~ 1 + cdar(week; by=patch, cor=C)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__, total_groups=())
    plan, names = sb_parity(model)
    cells = [c for c in plan.cells if c.family === :cdar]
    @test length(cells) == length(plan.cells) == 4
    @test all(c -> c.scale.cdiag ≈ 1.0, cells)
end

@stestset "SBBRMI: HSGP cells at their compiled centeredness, and grouped levels" begin
    data = merge(hsgp_data(; seed=35), (; hsgp_c=[0.0, 0.35, 0.75, 1.0]))
    model = SBBRMI(@brm(data, begin
        mu ~ 1 + hsgp(x; k=4, c=1.5, centeredness=hsgp_c) + (1 | u)
        y ~ Normal(mu, 0.2)
    end); mod=@__MODULE__, total_groups=())
    plan, names = sb_parity(model)
    weights = [c for c in plan.cells if c.family === :hsgp]
    @test [c.target for c in weights] == [0.0, 0.35, 0.75, 1.0]
    @test count(c -> c.family === :ranef, plan.cells) == 5
    model = SBBRMI(@brm(data, begin
        mu ~ 1 + hsgp(x; k=3, by=g)
        y ~ Normal(mu, 0.2)
    end); mod=@__MODULE__, total_groups=())
    plan, names = sb_parity(model)
    @test length(plan.cells) == 9 && all(c -> c.family === :hsgp, plan.cells)
    @test [b.level_index for b in plan.blocks] == 1:3
end

@stestset "SBBRMI: S2Z contrasts beside ordinary cells" begin
    data = (; band = repeat(["b1", "b2", "b3"], 4), h = repeat(["h1", "h2"], inner=6),
              y = randn(Xoshiro(3), 12))
    model = SBBRMI(merge(@brm(begin
        mu ~ 1 + (1 | band) + (1 | h)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(data); mod=@__MODULE__, total_groups=(), s2z_groups=[:band], s2z_rho=1.0)
    plan, names = sb_parity(model)
    @test [c.family for c in plan.cells] == [:ranef, :ranef, :scalar, :scalar]
    @test only(b for b in plan.blocks if b.family === :scalar).kind === :s2z
end

function level_models()
    data = merge(hsgp_data(; seed=39), (; C=[1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]))
    [@brm(data, begin
         Omega ~ LKJCholesky(3, 2.0)
         mu ~ 1 + hsgp(x; k=2, by=g, cor=Omega) + hsgp(x2; k=2) + (1 | u)
         log(sd(hsgp(x))) ~ 1 + (1 | g)
         log(length_scale(hsgp(x))) ~ 1 + (1 | g)
         y ~ Normal(mu, 0.3)
     end),
     @brm(data, begin
         mu ~ 1 + hsgp(x, x2; k=(2, 2), by=g, cor=C, iso=false)
         y ~ Normal(mu, 0.3)
     end),
     @brm(data, begin
         mu ~ 1 + hsgp(x; k=2, by=g)
         log(sd(hsgp(x))) ~ 1 + (1 | g)
         length_scale(mu, hsgp(x)) = 0.8
         y ~ Normal(mu, 0.3)
     end)]
end

function check_level_rows(plan)
    levels = [b for b in plan.blocks if b.family === :hsgp && b.kind === :levels]
    b = only(levels)
    cells = [c for c in plan.cells if c.block == b.block]
    @test [c.level_index for c in cells] == repeat(1:b.n_levels, b.n_basis)
    @test [c.term for c in cells] == repeat(1:b.n_basis; inner=b.n_levels)
    @test all(c -> c.target == 0.0, cells)
    if any(c -> c.family === :ranef, plan.cells)
        @test first(cells).cell > maximum(c.cell for c in plan.cells if c.family === :ranef)
        @test first(cells).cell > maximum(c.cell for c in plan.cells if c.family === :hsgp &&
                                         plan.blocks[c.block].kind === :basis)
    end
end

@stestset "RKBRMI: grouped HSGP level rows" begin
    for brmi in level_models()
        _, plan, _ = rk_parity(brmi; hsgp_levels=true)
        check_level_rows(plan)
    end
    # The level switch is independent of the basis switch.
    _, plan, _ = rk_parity(first(level_models()); hsgp=false, hsgp_levels=true)
    @test any(c -> c.family === :hsgp, plan.cells)
    @test all(b -> b.family !== :hsgp || b.kind === :levels, plan.blocks)
end

# A model whose value statements combine its formulas (`mu = level + surface +
# course`) plans an `_RKValuePlan`, which keeps the formulas in its `regression`
# plan. Its HSGP terms (a basis vector read from the data, a floored length
# scale, a grouped term orthogonal to its line), a level-mapped grouped term and
# a car field describe and map as a formula model's do (snag
# rk-hsgp-adaptive-cf312e88: the term lookup read only a formula plan's
# predictors and raised `type _RKValuePlan has no field predictors`).
@stestset "RKBRMI: HSGP, level and car cells on a value plan" begin
    value_plan(rk) = rk.plan isa BayesianRegressionModels._RKValuePlan
    data = merge(hsgp_data(; seed=35), (; basis=[3, 2]))
    rk, plan, _ = rk_parity(@brm data begin
        surface ~ 0 + hsgp(x, x2; k=basis, c=1.5)
        course ~ 0 + hsgp(t; k=3, c=1.5, by=g, orthogonal_to=:linear)
        level ~ 1 + (1 | u)
        mu = level + surface + course
        length_scale(surface, hsgp(x, x2)) ~
            truncated(LogNormal(0.0, 0.7); lower=length_scale_floor)
        sigma ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end)
    @test value_plan(rk)
    @test [c.family for c in plan.cells] == vcat(fill(:ranef, 5), fill(:hsgp, 6 + 9))
    # (Not named `hsgp`: a local of that name shadows `hsgp` in the `@brm` blocks below.)
    hsgp_blocks = [b for b in plan.blocks if b.family === :hsgp]
    @test sort([(b.binding, b.n_basis) for b in hsgp_blocks]) ==
          [(:course, 3), (:course, 3), (:course, 3), (:surface, 6)]
    # The floored length scale is a bounded sampled coordinate (lower > 0).
    surface = [c for c in plan.cells if c.family === :hsgp && c.binding === :surface]
    rho = only(first(surface).scale.length_scales)
    @test rho.coordinate > 0 && rho.lower > 0
    # A grouped term with correlated levels, mapped across them.
    rk, plan, _ = rk_parity(@brm(data, begin
        Omega ~ LKJCholesky(3, 2.0)
        field ~ 0 + hsgp(x; k=2, by=g, cor=Omega)
        mu = 0.5 + field
        y ~ Normal(mu, 0.3)
    end); hsgp_levels=true)
    @test value_plan(rk)
    check_level_rows(plan)
    # A car field (`fields = true`), one cell per area as on a formula model.
    rk, plan, _ = rk_parity(@brm(car_data(), begin
        field ~ 1 + car(area; edges=(from, to), type=:bym2)
        mu = 0.5 + field
        y ~ Normal(mu, 1.0)
    end); fields=true)
    @test value_plan(rk)
    @test count(c -> c.family === :car, plan.cells) == 8
end

@stestset "SBBRMI: grouped HSGP level rows" begin
    for brmi in level_models()
        plan, _ = sb_parity(SBBRMI(brmi; mod=@__MODULE__, total_groups=()))
        check_level_rows(plan)
    end
end

@stestset "SBBRMI: exact totals, column designs and mixed families" begin
    rng = Xoshiro(40)
    g = repeat(1:6; inner=6)
    h = repeat(1:3, 12)
    x = randn(rng, length(g))
    w = [0., 1., 0.5, -0.3, 0.1, 0.8][g]
    data = (; g, h, x, w, xw=x .* w, y=randn(rng, length(g)))
    for (brmi, widths) in (
            (@brm(data, begin
                mu ~ 1 + x + (1 | g)
                y ~ Normal(mu, 0.7)
            end), [1]),
            (@brm(data, begin
                mu ~ 1 + w + x + xw + (1 | a | g) + (0 + x | b | g)
                y ~ Normal(mu, 0.7)
            end), [2, 2]),
            (@brm(data, begin
                mu ~ 1 + (1 | h) + hsgp(x; k=2)
                log(sigma) ~ 1 + (1 | g)
                y ~ Normal(mu, sigma)
            end), [1]))
        model = SBBRMI(brmi; mod=@__MODULE__, total_groups=[:g])
        plan, _ = sb_parity(model)
        totals = [b for b in plan.blocks if b.family === :scalar && b.kind === :total]
        @test length.(getproperty.(totals, :design)) == widths
        @test all(b -> length(b.cells) == 6, totals)
        @test all(c -> c.target == 1.0, (c for c in plan.cells if c.family === :scalar))
        @test minimum(first(b.cells) for b in totals) >
              maximum((c.cell for c in plan.cells if c.family !== :scalar); init=0)
    end
end

@stestset "SBBRMI: S2Z endpoint and group frames" begin
    data = (; g=repeat(1:4; inner=4), x=repeat([-0.9, -0.2, 0.5, 0.8], 4),
              y=randn(Xoshiro(41), 16))
    brmi = @brm data begin
        mu ~ 1 + x + (1 + x || g)
        y ~ Normal(mu, 0.6)
    end
    targets = reshape(collect(range(0.0, 1.0; length=8)), 4, 2)
    for (kwargs, expected) in (( (; s2z_rho=[0.0, 1.0]), vcat(zeros(3), ones(3))),
                               ((; s2z_coordinates=:groups, s2z_rho=targets), vec(targets)))
        model = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), s2z_groups=[:g], kwargs...)
        plan, names = sb_parity(model)
        @test [c.target for c in plan.cells] == expected
        @test all(c -> c.family === :scalar && c.location.constant == 0.0, plan.cells)
        if !haskey(kwargs, :s2z_coordinates)
            interior = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), s2z_groups=[:g], s2z_rho=0.4)
            @test_throws r"interior centering weights" adaptive_centering_cells(interior, names)
        end
    end
end

@stestset "SBBRMI: exact totals precede S2Z cells" begin
    data = (; g=repeat(1:6; inner=6), h=repeat(1:3, 12),
              y=randn(Xoshiro(42), 36))
    brmi = @brm data begin
        mu ~ 1 + (1 | g)
        log(sigma) ~ 1 + (1 | h)
        y ~ Normal(mu, sigma)
    end
    model = SBBRMI(brmi; mod=@__MODULE__, total_groups=[:g], s2z_groups=[:h], s2z_rho=1.0)
    plan, _ = sb_parity(model)
    @test [b.kind for b in plan.blocks] == [:total, :s2z]
    @test [c.block for c in plan.cells] == vcat(fill(1, 6), fill(2, 2))
end
