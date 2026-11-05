# RUN: julia --project=test test/component_formula_starts.jl
# RUN: julia --project=test test/component_formula_starts.jl --stan
#
# `component_inits` takes the top-level `inits` grammar for a component call's
# own linear predictors and `|ID|` blocks (snag consume-the-deli-87327afc):
# link-scale coefficients, a shared block's scales, correlation, covariance and
# per-level effects, in repeated and nested calls on different row axes. The
# default run builds RK models only; `--stan` adds compiled Stan readback of
# the calls' predictor values and cross-backend physical-point parity.
using Test, BayesianRegressionModels, Distributions, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels

@brm function linked_rates(unit)
    log(rise) ~ 1 + (1 | rates | unit)
    log(fall) ~ 1 + (1 | rates | unit)
    effect(rise, Intercept) ~ Normal(log(1 / 6), .4)
    effect(fall, Intercept) ~ Normal(log(.2), .3)
    sd(rise, rates) ~ Exponential(1)
    sd(fall, rates) ~ Exponential(2)
    cor(:, rates) ~ LKJCholesky(2, 2)
    return rise + fall
end

@brm function wrapped_rates(unit)
    inner ~ linked_rates(unit)
    return inner
end

# A formula call whose scalar prior reads a parameter-valued input: traced on
# its own, `center` is a keyword it does not hold.
@brm function shifted_rates(unit, center)
    log(rise) ~ 1 + (1 | rates | unit)
    effect(rise, Intercept) ~ Normal(-1.0, .4)
    sd(rise, rates) ~ Exponential(1)
    a ~ Normal(center, .2)
    return rise + a
end
const SHIFTED_BUILDER = @brm begin
    left ~ shifted_rates(unit_l, center)
    y_l ~ Normal(left, sigma)
    center ~ Normal(.3, .4)
    sigma ~ Exponential(1)
end
const SHIFTED_DATA = (; unit_l=["a", "a", "b", "c"], y_l=[.3, .4, .2, .5])
const SHIFTED_COMPONENTS = [:left => (; rise=(; Intercept=.4), rates=(; sd=.3, effects=["a" => .2]))]

const FORMULA_DATA = (; unit_l=["a", "a", "b", "c"], unit_r=["u", "v", "v"],
    unit_n=["p", "q", "p", "r", "s"], g=["x", "y", "x", "y"],
    y_l=[.3, .4, .2, .5], y_r=[.1, .6, .2], y_n=[.2, .3, .1, .4, .5], z=[.1, .2, .3, .4])
const FORMULA_BUILDER = @brm begin
    left ~ linked_rates(unit_l)
    right ~ linked_rates(unit_r)
    nested ~ wrapped_rates(unit_n)
    mu ~ 1 + (1 | g)
    y_l ~ Normal(left, sigma)
    y_r ~ Normal(right, sigma)
    y_n ~ Normal(nested, sigma)
    z ~ Normal(mu, sigma)
    sigma ~ Exponential(1)
end

const SIGMA_N = [.25 .1; .1 .64]          # covariance of the nested call's block
const FORMULA_TOP = (; sigma=.6, mu=(; Intercept=.2))
const FORMULA_COMPONENTS = [
    :left => (; rise=(; Intercept=log(1 / 6)), fall=(; Intercept=log(.2)),
        rates=(; sd=[.3, .5], cor=[1 .4; .4 1], effects=["a" => [.1, -.2], "c" => [.05, .3]])),
    :right => (; fall=(; Intercept=.7), rates=(; sd=.4)),
    (:nested, :inner) => (; rise=(; Intercept=-1.0), rates=SIGMA_N),
]

# The natural effects of one call's `|rates|` block, levels × (rise, fall), from
# its constrained scales, correlation factor and standardized draws.
block_of(values) = getproperty(values,
    only(k for k in keys(values) if startswith(string(k), "b_")))
function block_parts(values)
    b = block_of(values)
    tau = hasproperty(b, :tau) ? collect(b.tau) : [b.tau_1, b.tau_2]
    L = Matrix(b.L)
    (; tau, L, z=b.z, effects=b.z * transpose(Diagonal(tau) * L))
end
coefficient(values, predictor) = getproperty(values, Symbol(predictor, :_Intercept))

@testset "component formula and shared-block starts (RK)" begin
    saved = deepcopy((FORMULA_DATA, FORMULA_TOP, FORMULA_COMPONENTS))
    rk = RKBRMI(FORMULA_BUILDER(FORMULA_DATA))
    n = rk.model.layout.total
    base = collect(range(-.3, .4; length=n))
    base_saved = copy(base)
    before = BRM._rk_constrained_values(rk, base)
    u = brm_start(rk, FORMULA_TOP; component_inits=FORMULA_COMPONENTS, base)
    v = BRM._rk_constrained_values(rk, u)
    @test base == base_saved
    @test isequal((FORMULA_DATA, FORMULA_TOP, FORMULA_COMPONENTS), saved)

    # Top-level values and the top-level block are untouched by component starts.
    @test v.sigma ≈ .6
    @test coefficient(v, :mu) ≈ .2
    @test block_of(v) == block_of(before)

    left = block_parts(v.left)
    @test coefficient(v.left, :rise) ≈ log(1 / 6)
    @test coefficient(v.left, :fall) ≈ log(.2)
    @test left.tau ≈ [.3, .5]
    @test left.L * left.L' ≈ [1 .4; .4 1]
    # Levels a and c take their effects; b keeps its base draws.
    @test left.effects[1, :] ≈ [.1, -.2]
    @test left.effects[3, :] ≈ [.05, .3]
    @test left.z[2, :] == block_parts(before.left).z[2, :]

    right = block_parts(v.right)
    @test coefficient(v.right, :fall) ≈ .7
    @test coefficient(v.right, :rise) == coefficient(before.right, :rise)
    @test right.tau ≈ [.4, .4]
    @test right.L == block_parts(before.right).L

    inner = block_parts(v.nested.inner)
    @test coefficient(v.nested.inner, :rise) ≈ -1.0
    @test Diagonal(inner.tau) * inner.L * inner.L' * Diagonal(inner.tau) ≈ SIGMA_N

    # A prior-median base fills each call's coefficients and scales from its
    # own priors; explicit values keep precedence.
    prior = BRM._rk_constrained_values(rk,
        brm_start(rk, FORMULA_TOP; component_inits=FORMULA_COMPONENTS, base=:prior_median))
    @test coefficient(prior.left, :rise) ≈ log(1 / 6)
    @test coefficient(prior.right, :rise) ≈ log(1 / 6)
    @test coefficient(prior.right, :fall) ≈ .7
    @test coefficient(prior.nested.inner, :fall) ≈ log(.2)
    @test block_parts(prior.right).tau ≈ [.4, .4]
    @test block_parts(prior.nested.inner).tau ≈ sqrt.(diag(SIGMA_N))
    plain = BRM._rk_constrained_values(rk, brm_start(rk, (;); base=:prior_median))
    @test coefficient(plain.left, :fall) ≈ log(.2)
    @test block_parts(plain.left).tau ≈ [median(Exponential(1)), median(Exponential(2))]
    @test block_parts(plain.nested.inner).tau ≈ [median(Exponential(1)), median(Exponential(2))]

    for bad in ([:left => (; rise=.3)],                       # a predictor is computed
                [:left => (; rise=(; x=.1))],                 # no such coefficient
                [:left => (; rates=(; sd=[.3]))],             # one scale per margin
                [:left => (; rates=(; sd=[-.3, .2]))],
                [:left => (; rates=(; cor=[1 2; 2 1]))],
                [:left => (; rates=(; effects=["zz" => [.1, .2]]))],  # not a level of this call
                [:left => (; rates=(; effects=["a" => [.1]]))],
                [:right => (; rates=(; effects=["a" => [.1, .2]]))],  # left's level, not right's
                [:left => (; rates=[1. 0; 0 -1])])
        @test_throws Exception brm_start(rk, (;); component_inits=bad)
    end
    @test_throws r"component \(:left,\)" brm_start(rk, (;);
        component_inits=[:left => (; rates=(; effects=["zz" => [.1, .2]]))])
end

if "--stan" in ARGS
    using BridgeStan
    # The component blocks' coordinates are not in `brm_coordinate_transport`'s
    # inventory, so parity is checked on physical values: each call's predictor
    # values per row (intercept plus its level's effect, through the link) and
    # its block's scales and correlation.
    rk_levels(column) = sort(unique(column))
    rk_rows(values, predictor, column) = exp.(coefficient(values, predictor) .+
        block_parts(values).effects[indexin(column, rk_levels(column)),
            predictor === :rise ? 1 : 2])
    function stan_value(θ, names, name)
        i = findfirst(==(name), names)
        i === nothing && error("no Stan value $name among " *
            join(filter(n -> occursin("rates", n), names), ", "))
        θ[i]
    end
    stan_block(θ, names, stem) = (;
        tau=[stan_value(θ, names, "$(stem)_tau.$k") for k in 1:2],
        cor=let L = [stan_value(θ, names, "$(stem)_L.$i.$j") for i in 1:2, j in 1:2]
            L * L'
        end)
    @testset "component formula starts: compiled Stan readback and RK parity" begin
        model = FORMULA_BUILDER(FORMULA_DATA)
        rk = RKBRMI(model)
        sb = SBBRMI(model; mod=@__MODULE__, total_groups=())
        stan = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "component-formula.stan")).model
        names = String.(BridgeStan.param_names(stan; include_tp=true))
        d = brm_descriptor(sb)
        function predictor(u, path, name)
            values = BridgeStan.param_constrain(stan, u; include_tp=true)
            values[brm_output_coordinates(d, name, names; submodel=path)]
        end
        calls = ((:left, "left_b_rates_unit", FORMULA_DATA.unit_l, v -> v.left),
            (:right, "right_b_rates_unit", FORMULA_DATA.unit_r, v -> v.right),
            ((:nested, :inner), "nested_inner_b_rates_unit", FORMULA_DATA.unit_n,
                v -> v.nested.inner))
        for base in (zeros(rk.model.layout.total), :prior_median)
            r = BRM._rk_constrained_values(rk,
                brm_start(rk, FORMULA_TOP; component_inits=FORMULA_COMPONENTS, base))
            s = brm_start(sb, stan, FORMULA_TOP;
                component_inits=FORMULA_COMPONENTS, base=base isa Symbol ? base :
                    zeros(BridgeStan.param_unc_num(stan)))
            @test isfinite(BridgeStan.log_density(stan, s))
            θ = BridgeStan.param_constrain(stan, s; include_tp=true)
            @test stan_value(θ, names, "sigma") ≈ r.sigma
            for (path, stem, column, local_values) in calls, p in (:rise, :fall)
                @test predictor(s, path, p) ≈ rk_rows(local_values(r), p, column)
            end
            for (path, stem, column, local_values) in calls
                rb, sb_block = block_parts(local_values(r)), stan_block(θ, names, stem)
                @test sb_block.tau ≈ rb.tau
                @test sb_block.cor ≈ rb.L * rb.L'
            end
            base isa AbstractVector || continue
            # At a zero base unlisted levels have zero effects, so every row's
            # predictor is exp(intercept + its level's given effect).
            rows_l = Dict("a" => [.1, -.2], "b" => [0., 0.], "c" => [.05, .3])
            @test predictor(s, :left, :rise) ≈
                [exp(log(1 / 6) + rows_l[u][1]) for u in FORMULA_DATA.unit_l]
            @test predictor(s, :left, :fall) ≈
                [exp(log(.2) + rows_l[u][2]) for u in FORMULA_DATA.unit_l]
            @test predictor(s, :right, :fall) ≈ fill(exp(.7), 3)
            @test predictor(s, (:nested, :inner), :rise) ≈ fill(exp(-1.0), 5)
            @test stan_block(θ, names, "left_b_rates_unit").tau ≈ [.3, .5]
            @test stan_block(θ, names, "left_b_rates_unit").cor ≈ [1 .4; .4 1]
            inner = stan_block(θ, names, "nested_inner_b_rates_unit")
            @test Diagonal(inner.tau) * inner.cor * Diagonal(inner.tau) ≈ SIGMA_N
        end
        for bad in ([:left => (; rise=.3)], [:left => (; rates=(; sd=[.3]))],
                [:right => (; rates=(; effects=["a" => [.1, .2]]))])
            @test_throws Exception brm_start(sb, stan, (;); component_inits=bad)
        end
    end
    @testset "component formula starts: a call whose prior reads a parameter input" begin
        model = SHIFTED_BUILDER(SHIFTED_DATA)
        rk = RKBRMI(model)
        sb = SBBRMI(model; mod=@__MODULE__, total_groups=())
        stan = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "component-shifted.stan")).model
        names = String.(BridgeStan.param_names(stan; include_tp=true))
        d = brm_descriptor(sb)
        s = brm_start(sb, stan, (;); component_inits=SHIFTED_COMPONENTS, base=:prior_median)
        θ = BridgeStan.param_constrain(stan, s; include_tp=true)
        read(name) = θ[brm_output_coordinates(d, name, names; submodel=:left)]
        effects = Dict("a" => .2, "b" => 0., "c" => 0.)
        @test read(:rise) ≈ [exp(.4 + effects[u]) for u in SHIFTED_DATA.unit_l]
        @test only(read(:a)) ≈ .3             # median(Normal(median(center), .2))
        @test stan_value(θ, names, "center") ≈ .3
        r = BRM._rk_constrained_values(rk,
            brm_start(rk, (;); component_inits=SHIFTED_COMPONENTS, base=:prior_median))
        @test coefficient(r.left, :rise) ≈ .4
        @test r.left.a ≈ only(read(:a))
        @test r.center ≈ .3
    end
end
