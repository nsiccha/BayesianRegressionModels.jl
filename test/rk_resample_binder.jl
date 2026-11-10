# test/rk_resample_binder.jl — a draw of the fit on designs that resample
# groupings (snag `rk-replay-with-r-bf059724`, decision `1ofewg8`).
#
# `rk_design_binder(rk; resample_groups)` gives the inputs of a cut at a draw
# `u` of the fit, with the new levels' effects in HAVE. The design's own binding
# sizes a resampled block's standardized draws by the design's levels, and RKPPL
# unpacks `u` with the binding's layout; the binder's inputs keep the fit's
# level set, so `u` is read in the fit's layout for fewer, as many and more new
# levels than the fit has. Oracles are independent: the fit's own graph values
# at the subjects a design copies, given those subjects' fitted effects, and
# closed-form linear predictors. Every coordinate of the draw is distinct, so a
# misplaced coordinate cannot go unnoticed (the reporter's draw had them all
# distinct; a constant draw hides the shift). Public synthetic data only.
#
# Run: julia --project=test test/rk_resample_binder.jl
using Test
include(joinpath(@__DIR__, "testset_filter.jl"))
using BayesianRegressionModels, Distributions, Random, Statistics
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels

module ResampleModels
using BayesianRegressionModels, Distributions
weighted(d) = @brm d begin
    sigma ~ Exponential(1)
    lp ~ 1 + zscale(weight) + (1 | subject)
    effect(lp, Intercept) ~ Normal(1.6, 0.5)
    y ~ Normal(lp, sigma)
end
correlated(d) = @brm d begin
    mu ~ 1 + x + (1 + x | p | subject)
    log(sigma) ~ 1 + (1 | p | subject)
    y ~ Normal(mu, sigma)
end
membership(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | mm(g1, g2))
    y ~ Normal(mu, sigma)
end
stratified(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | gr(g, by = s))
    y ~ Normal(mu, sigma)
end
end
const RM = ResampleModels

const N = 12
const WEIGHTS = (; subject=collect(1:N), weight=50 .+ 40 .* rand(Xoshiro(1), N),
    y=randn(Xoshiro(2), N))
const CORRELATED = (; subject=repeat(["s$k" for k in 1:6], inner=3),
    x=collect(range(-1.0, 1.5; length=18)), y=sin.(1:18))
const MEMBERS = (; g1=["a", "b", "c", "a", "d"], g2=["b", "c", "a", "d", "c"], y=randn(Xoshiro(3), 5))
const STRATA = (; g=["a", "a", "b", "b", "c", "c"], s=["u", "u", "u", "u", "v", "v"],
    y=randn(Xoshiro(4), 6))

distinct(rk) = collect(range(-0.5, 0.5; length=rk.model.layout.total))
names_of(rk) = ReactiveKernelsPPL.coordinate_names(rk.model.layout)
only_value(v) = v isa Tuple ? only(v) : v

# A cut of `rk`'s graph at a draw of the fit, the block `effects` in HAVE.
function cut_value(rk, inputs, want, u, block, effects)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, block, keys(inputs)...), want=(want,), bound=inputs)
    only_value(Base.invokelatest(query, u, effects))
end
# The binder recipe of brm-use: the fitted inputs bound once, the rest per design.
function binder_value(rk, binder, design, want, u, block, effects)
    inputs = binder(design)
    rest = Tuple(n for n in keys(inputs) if !haskey(binder.fitted, n))
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, block, rest..., keys(binder.fitted)...), want=(want,),
        bound=binder.fitted)
    only_value(Base.invokelatest(query, u, effects, (inputs[n] for n in rest)...))
end
# A value of `rk`'s graph on its own data at `u`.
function own_value(rk, want, u)
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=(want,), bound)
    only_value(Base.invokelatest(query, u))
end

@stestset "fewer, as many and more new subjects than the fit, the reporter's shape" begin
    rk = RKBRMI(RM.weighted(WEIGHTS))
    u = distinct(rk)
    names = names_of(rk)
    u[findfirst(==(Symbol("pop_lp.beta_pop.1")), names)] = 1.6
    u[findfirst(==(Symbol("pop_lp.beta_pop.2")), names)] = 0.3
    z(w) = (w .- mean(WEIGHTS.weight)) ./ std(WEIGHTS.weight)
    binder = rk_design_binder(rk; resample_groups=[:subject])
    fit_levels = rk_bound_data(rk).subject_fit_levels
    @test binder.fitted.subject_fit_levels == fit_levels
    # One kind of design: the first replays in full, the later ones bind into
    # its lowering, with any number of new subjects.
    for m in (2, 12, 20, 1, 7)
        w = 50 .+ 40 .* rand(Xoshiro(10 + m), m)
        design = (; subject=collect(100 .+ (1:m)), weight=w, y=zeros(m))
        inputs = binder(design)
        replay = rk_bound_data(reprocess(rk, design; resample_groups=[:subject]))
        @test keys(inputs) == keys(replay)
        @test inputs.subject_fit_levels == fit_levels
        @test all(isequal(inputs[n], replay[n]) for n in keys(inputs) if n !== :subject_fit_levels)
        @test inputs.subject_level == 1:m
        expected = 1.6 .+ 0.3 .* z(w)
        @test cut_value(rk, inputs, :lp, u, :b_subject, zeros(m, 1)) ≈ expected atol=1e-12
        b = reshape(collect(1.0:m), m, 1)
        @test cut_value(rk, inputs, :lp, u, :b_subject, b) ≈ expected .+ vec(b) atol=1e-12
        @test binder_value(rk, binder, design, :lp, u, :b_subject, b) ≈
            expected .+ vec(b) atol=1e-12
    end
    # Twins of fitted subjects under their own labels, given their fitted
    # effects, are the fit's own values at those rows.
    lp, effects = own_value(rk, :lp, u), own_value(rk, :b_subject, u)
    twins = (; subject=[7, 2], weight=WEIGHTS.weight[[7, 2]], y=zeros(2))
    @test cut_value(rk, binder(twins), :lp, u, :b_subject, effects[[2, 7], :]) ≈
        lp[[7, 2]] atol=1e-12
end

@stestset "a correlated block shared by two predictors" begin
    rk = RKBRMI(RM.correlated(CORRELATED))
    u = distinct(rk)
    mu, sigma, effects = own_value(rk, :mu, u), own_value(rk, :sigma, u),
        own_value(rk, :b_p_subject, u)
    binder = rk_design_binder(rk; resample_groups=[:subject])
    # Subjects s5, s1 and s3 under new labels (in that level order), their rows
    # and fitted effects: the fit's values at their rows.
    for keep in (["s5", "s1"], ["s5", "s1", "s3"], ["s2", "s4", "s6", "s1", "s3", "s5", "s2", "s4"])
        rows = reduce(vcat, [findall(==(s), CORRELATED.subject) for s in keep])
        labels = repeat(["n" * lpad(k, 2, '0') for k in eachindex(keep)], inner=3)
        design = (; subject=labels, x=CORRELATED.x[rows], y=zeros(length(rows)))
        level = [parse(Int, s[2:end]) for s in keep]
        inputs = binder(design)
        @test cut_value(rk, inputs, :mu, u, :b_p_subject, effects[level, :]) ≈ mu[rows] atol=1e-12
        @test cut_value(rk, inputs, :sigma, u, :b_p_subject, effects[level, :]) ≈ sigma[rows] atol=1e-12
    end
end

@stestset "a multi-membership block" begin
    rk = RKBRMI(RM.membership(MEMBERS))
    u = distinct(rk)
    intercept = ReactiveKernelsPPL.constrain(rk.model.layout, u).mu_Intercept
    binder = rk_design_binder(rk; resample_groups=[:g1, :g2])
    @test haskey(binder.fitted, :b_mm__g1__g2_groups)
    for (g1, g2) in ((["x", "y"], ["y", "x"]), (["p", "q", "r", "s", "t", "u"], ["q", "w", "v", "s", "a", "z"]))
        levels = sort(unique(vcat(g1, g2)))
        b = reshape(collect(range(-1.0, 2.0; length=length(levels))), :, 1)
        position(l) = findfirst(==(l), levels)
        expected = intercept .+ (b[position.(g1)] .+ b[position.(g2)]) ./ 2
        design = (; g1, g2, y=zeros(length(g1)))
        @test cut_value(rk, binder(design), :mu, u, :b_mm__g1__g2, b) ≈ expected atol=1e-12
    end
end

@stestset "a data-free fit's binder takes the fitted state's level set" begin
    rk = RKBRMI(RM.weighted(WEIGHTS))
    u = distinct(rk)
    state = brm_fitted_state(rk)
    design = (; subject=collect(100 .+ (1:20)), weight=50 .+ 40 .* rand(Xoshiro(5), 20),
        y=zeros(20))
    # The brm-use recipe: a data-free replay of one design, then its binder.
    free = RKBRMI(RM.weighted(design), state; resample_groups=[:subject])
    from_state = rk_design_binder(free; resample_groups=[:subject])
    from_fit = rk_design_binder(rk; resample_groups=[:subject])
    @test from_state.fitted.subject_fit_levels == rk_bound_data(rk).subject_fit_levels
    b = reshape(collect(range(-2.0, 2.0; length=20)), 20, 1)
    for d in (design, merge(design, (; subject=[3, 4], weight=design.weight[1:2], y=zeros(2))))
        m = length(d.subject)
        @test cut_value(rk, from_state(d), :lp, u, :b_subject, b[1:m, :]) ≈
            cut_value(rk, from_fit(d), :lp, u, :b_subject, b[1:m, :]) atol=1e-12
    end
    # The data-free replay itself is a model of its own rows: its coordinates
    # hold the design's 20 subjects.
    @test count(n -> startswith(String(n), "b_subject.z."),
        BRM._rk_bound_coordinate_names(free)) == 20
end

@stestset "a gr(g, by = s) block keeps the fit's coordinates only at the fit's level counts" begin
    rk = RKBRMI(RM.stratified(STRATA))
    u = distinct(rk)
    binder = rk_design_binder(rk; resample_groups=[:g, :s])
    # Three groups over two strata, as fitted: the coordinates are the fit's.
    same = (; g=["x", "y", "z"], s=["p", "p", "q"], y=zeros(3))
    @test keys(binder(same)) == keys(rk_bound_data(reprocess(rk, same; resample_groups=[:g, :s])))
    # Two groups: the block's draws are sized by the design's own columns.
    fewer = (; g=["x", "y"], s=["p", "q"], y=zeros(2))
    @test_throws "other coordinates than the fit" binder(fewer)
    @test_throws "The `gr(...)` block over `g`, `s`" binder(fewer)
end
