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
# nested in a band (containing cells) and a cdar walk passing through; on
# SBBRMI, factor levels and crossed containing groupings, a Student-t block
# nested in an ordinary one beside a centred block with known SDs and
# correlation (the joint derived-scale wrapper), and an R2D2 block whose scales
# are unit and simplex atoms. An SBBRMI wrapper that also adapts S2Z contrasts
# raises.
#
#     julia --project=test test/adaptive_centering_cells.jl [testset filter...]
using Test, BayesianRegressionModels, Distributions, Random, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems, WarmupHMC
using DifferentiationInterface: AutoEnzyme
using CategoricalArrays
import StanBlocks
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
        acc += a.weight * log_simplex(x[a.coordinates], a.entry)
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

# Source `x` -> compiled `y` and the log-Jacobian, cell by cell in row order.
function public_map(x, plan, controls)
    T = eltype(x)
    y = copy(x)
    ljac = zero(T)
    s = zeros(T, length(plan.cells))
    loc = zeros(T, length(plan.cells))
    for c in plan.cells
        b = plan.blocks[c.block]
        C = block_factor(b, x, c.level_index)
        t, k, cc = b.target, c.term, controls[c.cell]
        m = zero(T)
        for (l, q) in enumerate(c.conditioning)
            m += C[k, l] * (y[plan.cells[q].effect] - t * loc[q]) / s[q]^t
        end
        L = m + (c.location.coefficient == 0 ? zero(T) : x[c.location.coefficient])
        for j in c.location.factors
            L += x[j]
        end
        for a in c.location.containers
            ya = y[plan.cells[a].effect]
            L += plan.blocks[plan.cells[a].block].target == 0 ? s[a] * ya : ya
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
function location_label(plan, c, names)
    parts = String[]
    c.location.coefficient == 0 || push!(parts, names[c.location.coefficient])
    append!(parts, names[c.location.factors])
    append!(parts, (plan.cells[a].name for a in c.location.containers))
    isempty(parts) ? nothing : join(parts, " + ")
end

function check_rows(plan, rows, names)
    @test plain(plan)
    @test [c.cell for c in plan.cells] == 1:length(plan.cells)
    @test [b.block for b in plan.blocks] == 1:length(plan.blocks)
    ranef = [r for r in rows if r.family === :ranef]
    @test length(ranef) == length(plan.cells) == length(rows)
    @test [r.index for r in ranef] == [c.effect for c in plan.cells]
    @test [r.name for r in ranef] == [c.name for c in plan.cells] ==
          [names[c.effect] for c in plan.cells]
    @test [r.binding for r in ranef] == [c.binding for c in plan.cells]
    @test [r.term for r in ranef] == [c.term for c in plan.cells]
    @test [r.margin for r in ranef] == [c.margin for c in plan.cells]
    @test [r.target for r in ranef] == [plan.blocks[c.block].target for c in plan.cells]
    @test [r.location for r in ranef] == [location_label(plan, c, names) for c in plan.cells]
    for c in plan.cells
        @test all(<(c.cell), c.location.containers)
        @test all(<(c.cell), c.conditioning)
        @test length(c.conditioning) == c.term - 1
    end
end

function check_parity(model, problem, names; trials=2, rng=Xoshiro(7), fd=true)
    plan = adaptive_centering_cells(model, names)
    n = length(plan.cells)
    check_rows(plan, adaptive_centering_controls(
        adaptive_centering_problem(model, problem, AD; unc_names=names), model;
        unc_names=names), names)
    d = length(names)
    for controls in (rand(rng, [0.0, 0.3, 0.7, 1.0], n),
                     [rand(rng, 0:0.1:1) for _ in 1:n])[1:trials]
        adaptive = adaptive_centering_problem(model, problem, AD; unc_names=names,
                                              centeredness=controls)
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

function rk_parity(brmi)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    # The RK layout's coordinate names (brm-use, "RK readers").
    names = String.(ReactiveKernelsPPL.coordinate_names(rk.model.layout))
    # The default `unc_names` of the RKBRMI method is that layout.
    default = adaptive_centering_cells(rk)
    plan = check_parity(rk, problem, names)
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
    @test all(c -> isempty(c.location.containers) && isempty(c.location.factors),
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

@stestset "RKBRMI: a cdar walk beside the block passes through" begin
    df = panel(; G=6, n=10, seed=9)
    data = merge(df, (; step=repeat(1:10, 6), I_g=Matrix(1.0I, 6, 6)))
    rk, plan, names = rk_parity(@brm data begin
        mu ~ 1 + t + (1 + t || g) + cdar(step; by = g, cor = I_g)
        effect(mu, Intercept) ~ Normal(100, 50)
        sd(:, g) ~ Exponential(30.0)
        y ~ Normal(mu, se)
    end)
    @test length(plan.cells) == 12
    @test !any(c -> occursin(".eta.", c.name), plan.cells)
end

# ---- SBBRMI -----------------------------------------------------------------

const BS = StanBlocks.BridgeStan
instantiate(model) = Base.invokelatest(StanBlocks.stan_instantiate, model.model)

function sb_parity(model; kwargs...)
    problem = instantiate(model)
    names = BS.param_unc_names(problem.model)
    check_parity(model, problem, names; kwargs...), names
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
    @test all(c -> length(c.location.factors) == 1 && c.location.coefficient == 0, pairs)
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
    @test all(c -> c.location == (; coefficient=0, factors=Int[], containers=Int[]),
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
end

@stestset "SBBRMI: a wrapper that also adapts S2Z contrasts raises" begin
    data = (; band = repeat(["b1", "b2", "b3"], 4), h = repeat(["h1", "h2"], inner=6),
              y = randn(Xoshiro(3), 12))
    model = SBBRMI(merge(@brm(begin
        mu ~ 1 + (1 | band) + (1 | h)
        y ~ Normal(mu, sigma)
    end), @brm(begin
        sigma ~ Exponential(1)
    end))(data); mod=@__MODULE__, total_groups=(), s2z_groups=[:band], s2z_rho=1.0)
    @test_throws r"S2Z contrasts" adaptive_centering_cells(model, String[])
end
