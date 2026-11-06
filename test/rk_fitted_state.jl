# test/rk_fitted_state.jl — a fitted RKBRMI's frozen preprocessing exported
# without its training rows, and designs bound from it alone (snag
# `no-data-free-exp-fec940cb`; the user picked "keep the fits in the RKPPL
# program and freeze them by value" on design decision `0f3hrtg`; RKPPL's
# `frozen_values` / `bind_data(...; frozen)` from ReactiveKernels snag
# `rkppl-bind-froze-da223711`).
#
# `brm_fitted_state(rk)` holds the values the emitted program fits from its
# fit-role inputs, the planner's Julia-side fits and the `constant(...)`
# values. Oracle: it holds no array of training-row length and round-trips
# through `Serialization`; planning a design from it alone emits the fit's
# program; and `RKBRMI(build(design), state)` binds the inputs, gives the graph
# values and the density that `reprocess(rk, design)` gives on the training
# fit, without binding a training column. Public synthetic models only.
#
# Run: julia --startup-file=no --project=test test/rk_fitted_state.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, Serialization, Statistics

module FittedStateModels
using BayesianRegressionModels, Distributions, Statistics
zscaled(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x)
    y ~ Normal(mu, sigma)
end
centered(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + center(x)
    y ~ Normal(mu, sigma)
end
standardized(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + standardize(x)
    y ~ Normal(mu, sigma)
end
zscaled_log(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(log(x))
    y ~ Normal(mu, sigma)
end
factors(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + g + gs + mo(r)
    y ~ Normal(mu, sigma)
end
grouped(d) = @brm d begin
    sigma ~ Exponential(1)
    xbar = constant(mean(x))
    mu ~ 1 + zscale(x) + (1 + z | subject)
    effect(mu, Intercept) ~ Normal(xbar, 1)
    y ~ Normal(mu, sigma)
end
smooth(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5)
    y ~ Normal(mu, sigma)
end
smooth_rows(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5, rows=keep)
    y ~ Normal(mu, sigma)
end
smooth_2d(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x, z; k=(3, 3))
    y ~ Normal(mu, sigma)
end
smooth_orthogonal(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + hsgp(x; k=5, orthogonal_to=:linear)
    y ~ Normal(mu, sigma)
end
smooth_orthogonal_domain(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + hsgp(x; k=5, domain=(0.0, 7.0), orthogonal_to=:linear)
    y ~ Normal(mu, sigma)
end
smooth_2d_centered(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x, z; k=(3, 3), orthogonal_to=:intercept)
    y ~ Normal(mu, sigma)
end
smooth_rows_centered(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5, rows=keep, orthogonal_to=:intercept)
    y ~ Normal(mu, sigma)
end
# `by=`: each level's projection is fitted on that level's rows.
smooth_grouped_centered(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x, z; k=(3, 3), by=gs, orthogonal_to=:intercept)
    y ~ Normal(mu, sigma)
end
smooth_grouped_linear(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + hsgp(x; k=5, by=gs, orthogonal_to=:linear)
    y ~ Normal(mu, sigma)
end
smooth_clamped(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5, extrapolate=:clamp)
    y ~ Normal(mu, sigma)
end
thin_plate(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + s(x)
    y ~ Normal(mu, sigma)
end
tensor(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + t2(x, z)
    y ~ Normal(mu, sigma)
end
shrunk(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + gs + (1 | subject)
    effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5, alpha=1)
    y ~ Normal(mu, sigma)
end
completed(d) = @brm d begin
    sigma ~ Exponential(1)
    mi(xm) ~ LogNormal(0.5, 0.6)
    mu ~ 1 + standardize(xm)
    y ~ Normal(mu, sigma)
end
end
const FM = FittedStateModels

const N = 24
const TRAIN = let rng = Random.Xoshiro(7)
    x = 1 .+ 4 .* rand(rng, N)
    subject = repeat(1:6, inner=4)
    g = categorical(repeat(["a", "b", "c"], 8))
    gs = repeat(["lo", "mid", "hi"], outer=8)
    r = repeat(1:4, outer=6)
    z = randn(rng, N)
    keep = repeat([1.0, 1.0, 0.0], 8)
    xm = Union{Missing,Float64}[i % 5 == 0 ? missing : exp(0.5 + 0.3 * v)
        for (i, v) in enumerate(randn(rng, N))]
    (; y=randn(rng, N), x, z, subject, g, gs, r, keep, xm)
end
const SUPPORTED = (FM.zscaled, FM.centered, FM.standardized, FM.zscaled_log,
    FM.factors, FM.grouped, FM.smooth, FM.smooth_rows, FM.smooth_2d,
    FM.smooth_orthogonal, FM.smooth_clamped, FM.smooth_orthogonal_domain,
    FM.smooth_2d_centered, FM.smooth_rows_centered, FM.smooth_grouped_centered,
    FM.smooth_grouped_linear, FM.tensor, FM.shrunk)
rows(data, index) = map(column -> column[index], data)
without(data, name) = NamedTuple{Tuple(k for k in keys(data) if k !== name)}(data)

# The design inputs a replay binds, without the fit-role inputs: the training
# columns on `reprocess`, which a data-free replay replaces with `state.fitted`.
function design_inputs(plan, emitted, state)
    columns = BRM._rk_source_data_columns(plan, emitted)
    Dict(name => value for (name, value) in columns if !(name in state.fit_inputs))
end

function check_planning(build, state, rk, design; resample_groups=())
    plan, emitted = BRM._rk_fitted_replay_plan(build(design), design, state; resample_groups)
    replay = reprocess(rk, design; resample_groups)
    @test BRM._rk_replay_statements(emitted, plan) == state.program
    @test isequal(design_inputs(plan, emitted, state),
        design_inputs(replay.plan, BRM._rk_retained_program(replay), state))
end

@stestset "the fitted values replace the training columns" begin
    for build in SUPPORTED
        rk = RKBRMI(build(TRAIN))
        state = brm_fitted_state(rk)
        # Something is fitted: in the program (an anchor, a level set read
        # through a fit-role input) or by the planner (a factor's levels).
        @test !isempty(state.fitted) || !isempty(state.fits)
        @test isempty(state.fitted) == isempty(state.fit_inputs)
        # No training rows travel: every fitted value is a summary, never a
        # column with one entry per training row.
        for value in Iterators.flatten((values(state.fitted.values), values(state.fits)))
            @test !(value isa AbstractArray) || length(value) != N
            @test !any(column -> isequal(value, column), values(TRAIN))
        end
        @test state.coordinates == BRM._rk_layout_coordinate_names(rk)
        io = IOBuffer()
        serialize(io, state)
        back = deserialize(seekstart(io))
        @test isequal(back.fitted, state.fitted)
        @test isequal(back.fits, state.fits)
        @test back.program == state.program
    end
end

@stestset "a fitted value is the training data's" begin
    state = brm_fitted_state(RKBRMI(FM.zscaled(TRAIN)))
    @test state.fitted.values.x_fit_mean ≈ mean(TRAIN.x)
    @test state.fitted.values.x_fit_sd ≈ std(TRAIN.x)
    state = brm_fitted_state(RKBRMI(FM.grouped(TRAIN)))
    @test state.fitted.values.subject_fit_levels == 1:6
    @test state.constants[:xbar] ≈ mean(TRAIN.x)
    state = brm_fitted_state(RKBRMI(FM.factors(TRAIN)))
    @test collect(state.fits[(:levels, :r)]) == 1:4
    # An automatic HSGP interval: the axis mean and the boundary-scaled
    # largest deviation from it (`c = 1.5` by default).
    state = brm_fitted_state(RKBRMI(FM.smooth(TRAIN)))
    center = mean(TRAIN.x)
    @test state.fitted.values.hsgp_x_fit[1] ≈ center
    @test state.fitted.values.hsgp_x_fit[2] ≈ 1.5 * maximum(abs.(TRAIN.x .- center))
end

@stestset "a design plans from the fitted state alone" begin
    for build in SUPPORTED
        rk = RKBRMI(build(TRAIN))
        state = brm_fitted_state(rk)
        check_planning(build, state, rk, rows(TRAIN, 3:9))
        check_planning(build, state, rk, rows(TRAIN, [5]))
        check_planning(build, state, rk, without(rows(TRAIN, 2:6), :y))
    end
    rk = RKBRMI(FM.grouped(TRAIN))
    state = brm_fitted_state(rk)
    newcomers = merge(rows(TRAIN, 1:8), (; subject=repeat([101, 102], inner=4)))
    check_planning(FM.grouped, state, rk, newcomers; resample_groups=[:subject])
end

@stestset "a fitted state keeps reprocess's refusals" begin
    rk = RKBRMI(FM.factors(TRAIN))
    state = brm_fitted_state(rk)
    # refused: an unseen level has no fitted effect (brm-use reactivekernels.md,
    # "Training levels only").
    unseen = merge(rows(TRAIN, 1:3), (; gs=["lo", "mid", "new"]))
    @test_throws "new" BRM._rk_fitted_replay_plan(FM.factors(unseen), unseen, state)
    # refused: a state of another model holds other fits, so the draws would
    # not mean the same thing.
    other = brm_fitted_state(RKBRMI(FM.zscaled(TRAIN)))
    @test_throws "exported from another model" BRM._rk_fitted_replay_plan(
        FM.centered(TRAIN), TRAIN, other)
end

# An authored value of `rk`'s graph on its own bound data.
function graph_value(rk, want, u)
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=(want,), bound)
    value = Base.invokelatest(query, u)
    value isa Tuple ? only(value) : value
end
point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
density(rk, u) = LogDensityProblems.logdensity(
    rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)
absent(v) = occursin("Absent", string(nameof(typeof(v))))

# The names a data-free binding of `rk`'s program holds absent: the fit-role
# inputs it drops, and the definitions fitted from those alone that no frozen
# value stands in for (`mu_gs_lvl_2_indicator`, read only by its frozen
# variance `mu_gs_lvl_2_indicator_var`).
function absent_names(rk, state, dropped, data)
    names = Set{Symbol}(dropped)
    for statement in BRM._rk_retained_program(rk).main.args
        Meta.isexpr(statement, :(=), 2) && statement.args[1] isa Symbol || continue
        reads = intersect(BRM._rk_source_symbols!(Set{Symbol}(), statement.args[2]), data)
        !isempty(reads) && issubset(reads, names) && push!(names, statement.args[1])
    end
    setdiff(names, keys(state.fitted.values))
end

# Data-free inputs `free` against a training replay's `replay` of the fit `rk`:
# the same values, except that a fit-role input and what is fitted from it
# alone are never bound (or bound absent), and a value RK preparation folds is
# bound frozen in place of the replay's graph statement. A resampled grouping's
# fit-role input holds the design's own values on both.
function check_inputs(free, replay, state, rk; resample_groups=())
    dropped = Set(input for input in state.fit_inputs
        if !(state.fit_twins[input] in resample_groups))
    unbound = absent_names(rk, state, dropped, Set{Symbol}(keys(replay)))
    for name in keys(free)
        if name in unbound
            @test absent(free[name])
        elseif haskey(replay, name)
            @test isequal(free[name], replay[name])
        else
            @test haskey(state.fitted.values, name)
        end
    end
    for name in keys(replay)
        name in dropped || @test haskey(free, name)
    end
end

function check_data_free(build, state, rk, design; resample_groups=(), want=:mu)
    free = RKBRMI(build(design), state; resample_groups)
    replay = reprocess(rk, design; resample_groups)
    @test free.fitted.state === state
    check_inputs(rk_bound_data(free), rk_bound_data(replay), state, rk; resample_groups)
    if isempty(resample_groups)
        @test BRM._rk_layout_coordinate_names(free) == state.coordinates
        u = point(replay)
        @test isequal(graph_value(free, want, u), graph_value(replay, want, u))
        hasproperty(design, :y) && @test density(free, u) == density(replay, u)
    end
    free
end

@stestset "data-free replay preserves scalar prior inputs" begin
    build = brm(quote
        sigma ~ Exponential(prior_scale)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end; mod=FM)
    train = (; x=[-2.0, 1.0, 4.0], y=[0.2, 0.7, -0.1], prior_scale=1.7)
    design = (; x=[2.0, 5.0], y=[0.3, -0.4])
    rk = RKBRMI(build(train))
    state = brm_fitted_state(rk)
    free = check_data_free(build, state, rk, design)
    @test state.scalars == Dict(:prior_scale => train.prior_scale)
    # Supplied values cannot change a fitted prior, including on a later replay.
    conflicting = merge(design, (; prior_scale=9.0))
    for fitted in (rk, free)
        @test_throws "changes the fitted RK program" reprocess(fitted, conflicting)
    end
    @test_throws "changes the fitted RK program" RKBRMI(build(conflicting), state)
    fresh = reprocess(rk, conflicting; freeze_constants=false)
    @test density(fresh, point(fresh)) == density(RKBRMI(build(conflicting)), point(fresh))
    io = IOBuffer()
    serialize(io, state)
    check_data_free(build, deserialize(seekstart(io)), rk, design)
    check_data_free(build, brm_fitted_state(free), rk, design)
    prediction = check_data_free(build, state, rk, (; x=design.x))
    check_data_free(build, brm_fitted_state(prediction), rk, design)
end

@stestset "a block prior's data scalar is an input of the regression program" begin
    # Snag `rk-sd-prior-data-fecc0c41`. A random-effect block reads its prior's
    # inputs by name. It takes the replay contract of
    # `sigma ~ Exponential(prior_scale)` above: a bound input, omitted or equal
    # on a replay, and refused when changed (user principle `0ikeidi`).
    build = brm(quote
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        sd(:, p) ~ Exponential(prior_sd_scale)
        y ~ Normal(mu, sigma)
    end; mod=FM)
    train = (; subject=[1, 1, 2, 2], x=[-2.0, 1.0, 4.0, 2.5], y=[0.2, 0.7, -0.1, 0.4],
        prior_sd_scale=1.7)
    design = (; subject=[1, 2], x=[2.0, 5.0], y=[0.3, -0.4])
    rk = RKBRMI(build(train))
    @test rk_bound_data(rk).prior_sd_scale == 1.7
    @test !(:(prior_sd_scale = 1.7) in BRM._rk_emit_ast(rk.plan).main.args)
    state = brm_fitted_state(rk)
    @test state.scalars == Dict(:prior_sd_scale => train.prior_sd_scale)
    free = check_data_free(build, state, rk, design)
    check_data_free(build, state, rk, (; subject=design.subject, x=design.x))
    # A held-out response and a build without the response read it the same way.
    joint = @brm merge(train, (; z=[0.1, 0.3, -0.2, 0.5])) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        eta ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(prior_sd_scale)
        y ~ Normal(mu, sigma)
        z ~ Normal(eta, sigma)
    end
    @test rk_bound_data(RKBRMI(joint; held_out=(:z,))).prior_sd_scale == 1.7
    predictive = RKBRMI(build((; subject=train.subject, x=train.x,
        prior_sd_scale=train.prior_sd_scale)))
    # With nothing observed, its program is empty and reads no input.
    @test !haskey(rk_bound_data(predictive), :prior_sd_scale)
    conflicting = merge(design, (; prior_sd_scale=9.0))
    for fitted in (rk, free)
        @test_throws "changes the fitted RK program" reprocess(fitted, conflicting)
    end
    @test_throws "changes the fitted RK program" RKBRMI(build(conflicting), state)
    fresh = reprocess(rk, conflicting; freeze_constants=false)
    @test density(fresh, point(fresh)) == density(RKBRMI(build(conflicting)), point(fresh))
    # The same value held by the design is the same program.
    equal = reprocess(rk, merge(design, (; prior_sd_scale=train.prior_sd_scale)))
    @test density(equal, point(equal)) == density(reprocess(rk, design), point(equal))
    binder = rk_design_binder(rk)
    for held in (design, merge(design, (; x=[1.0, 3.0])))
        inputs = binder(held)
        @test inputs.prior_sd_scale == 1.7
        @test Set(keys(inputs)) == Set(keys(rk_bound_data(reprocess(rk, held))))
    end
    @test !occursin("replayed in full", sprint(show, binder))

    # The default start reads the bound input: the median of the scale.
    model = brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    u = first(BRM._brm_nlme_prior_median_point(model))
    values = BRM._rk_constrained_values(rk, u)
    @test BRM._brm_rk_declaration_value(values, Symbol("b_p_subject.tau")) ≈
        fill(1.7log(2), 2)
end

@stestset "a term prior reading a data scalar is the model with its number" begin
    # The twin states the number; the literal forms have their own parity tests.
    data = (; x=[-2.0, 1.0, 4.0, 2.5, 0.3, -1.0], w=[0.1, 0.5, 0.3, 0.9, 0.2, 0.8],
        t=collect(1.0:6.0), y=[0.2, 0.7, -0.1, 0.4, 0.3, 0.1],
        subject=[1, 1, 2, 2, 3, 3], scale=1.7, rates=[1.7, 2.0, 0.5])
    area_data = (; area=[1, 2, 3, 1, 2, 3], y=data.y, from=[1, 2], to=[2, 3], half=0.5, low=0.2)
    panel_data = (; year=[1, 2, 3, 1, 2, 3], kreis=[1, 1, 1, 2, 2, 2], y=data.y,
        C=[1.0 0.0; 0.0 1.0], scale=1.7)
    field_data = (; east=[0.1, 0.9, 0.7, 0.3], north=[0.2, 0.4, 0.8, 0.5],
        time=[0.0, 0.3, 0.3, 1.7], y=[0.4, -0.3, 0.2, 0.1],
        xy=[0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5], triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5],
        scale=1.7, rates=[1.7, 2.0, 0.5])
    field_scale = @brm(field_data, begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(1.7)
        y ~ Normal(field, 0.2)
    end)
    twins = (
        "gmrf scale" => (@brm(field_data, begin
            field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east, north), time)) ~ Exponential(scale)
            y ~ Normal(field, 0.2)
        end), field_scale),
        "gmrf scale from a vector element" => (@brm(field_data, begin
            field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east, north), time)) ~ Exponential(rates[1])
            y ~ Normal(field, 0.2)
        end), field_scale),
        "gmrf scale from an assignment" => (@brm(field_data, begin
            rate = 1.7
            field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east, north), time)) ~ Exponential(rate)
            y ~ Normal(field, 0.2)
        end), field_scale),
        "gmrf scale from a constant" => (@brm(field_data, begin
            rate = constant(scale)
            field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east, north), time)) ~ Exponential(rate)
            y ~ Normal(field, 0.2)
        end), field_scale),
        "gmrf lengths" => (@brm(field_data, begin
            field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
            length_scale(field, gmrf((east, north), time), space) ~ LogNormal(scale, 0.8)
            length_scale(field, gmrf((east, north), time), time) ~ LogNormal(0.7, scale)
            y ~ Normal(field, 0.2)
        end), @brm(field_data, begin
            field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
            length_scale(field, gmrf((east, north), time), space) ~ LogNormal(1.7, 0.8)
            length_scale(field, gmrf((east, north), time), time) ~ LogNormal(0.7, 1.7)
            y ~ Normal(field, 0.2)
        end)),
        "gp scale" => (@brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 0 + gp(x, w; jitter=.03)
            sd(:, gp(x, w)) ~ Exponential(scale)
            y ~ Normal(mu, sigma)
        end), @brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 0 + gp(x, w; jitter=.03)
            sd(:, gp(x, w)) ~ Exponential(1.7)
            y ~ Normal(mu, sigma)
        end)),
        "dar scale" => (@brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + dar(t)
            sd(mu, dar(t)) ~ Normal(0, scale)
            y ~ Normal(mu, sigma)
        end), @brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + dar(t)
            sd(mu, dar(t)) ~ Normal(0, 1.7)
            y ~ Normal(mu, sigma)
        end)),
        "car scale and mixing share" => (@brm(area_data, begin
            mu ~ 1 + car(area; edges=(from, to), type=:bym2)
            sd(mu, car(area)) ~ Normal(0, half)
            mixing(mu, car(area)) ~ Uniform(low, 0.9)
            y ~ Normal(mu, 1.0)
        end), @brm(area_data, begin
            mu ~ 1 + car(area; edges=(from, to), type=:bym2)
            sd(mu, car(area)) ~ Normal(0, 0.5)
            mixing(mu, car(area)) ~ Uniform(0.2, 0.9)
            y ~ Normal(mu, 1.0)
        end)),
        "cdar scale and persistence" => (@brm(panel_data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + cdar(year; by=kreis, cor=C)
            sd(mu, cdar(year)) ~ Exponential(scale)
            ar(mu, cdar(year)) ~ Beta(scale, 2.0)
            y ~ Normal(mu, sigma)
        end), @brm(panel_data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + cdar(year; by=kreis, cor=C)
            sd(mu, cdar(year)) ~ Exponential(1.7)
            ar(mu, cdar(year)) ~ Beta(1.7, 2.0)
            y ~ Normal(mu, sigma)
        end)),
        "R2D2 reference scale" => (@brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | p | subject)
            sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1, reference_scale=scale)
            y ~ Normal(mu, sigma)
        end), @brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | p | subject)
            sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=2, concentration=1, reference_scale=1.7)
            y ~ Normal(mu, sigma)
        end)),
        "block scale from a vector element" => (@brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | p | subject)
            sd(:, p) ~ Exponential(rates[1])
            y ~ Normal(mu, sigma)
        end), @brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | p | subject)
            sd(:, p) ~ Exponential(1.7)
            y ~ Normal(mu, sigma)
        end)),
        "block scale from a vector reduction" => (@brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | p | subject)
            sd(:, p) ~ Exponential(maximum(rates) - 0.3)
            y ~ Normal(mu, sigma)
        end), @brm(data, begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | p | subject)
            sd(:, p) ~ Exponential(1.7)
            y ~ Normal(mu, sigma)
        end)))
    for (label, (read, stated)) in twins
        @testset "$label" begin
            a, b = RKBRMI(read), RKBRMI(stated)
            @test BRM._rk_layout_coordinate_names(a) == BRM._rk_layout_coordinate_names(b)
            # Not a design input: the binding may evaluate a data-only vector
            # definition, as it does a design matrix.
            @test isempty(intersect(
                keys(BRM._rk_source_data_columns(a.plan, BRM._rk_emit_ast(a.plan))),
                (:scale, :rates, :half, :low)))
            for u in (point(a), zeros(a.model.layout.total))
                @test density(a, u) ≈ density(b, u) rtol=1e-13
            end
        end
    end
    # Only numbers: anything else is refused in terms of the model.
    labels = merge(data, (; rates=["a", "b", "c"]))
    @test_throws "reads data field `rates`" RKBRMI(@brm(labels, begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | p | subject)
        sd(:, p) ~ Exponential(length(rates))
        y ~ Normal(mu, sigma)
    end))
end

@stestset "only a scalar the program reads by name gets a definition" begin
    data = (; x=[-2.0, 1.0, 4.0], y=[0.2, 0.7, -0.1], unused=5.0, scale=1.7)
    # A population prior folds its argument, and no statement reads the name.
    folded = @brm data begin
        sigma ~ Exponential(scale)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, scale)
        y ~ Normal(mu, sigma)
    end
    main = BRM._rk_emit_ast(BRM._brm_rk_plan(folded)).main
    @test !any(s -> Meta.isexpr(s, :(=)) && first(s.args) in (:scale, :unused), main.args)

    # Call heads, keyword names and declaration targets are not value reads, so
    # a data scalar named like a callee cannot shadow it.
    @test BRM._rk_value_reads!(Set{Symbol}(), :(b ~ f(g, s; k=v))) == Set([:g, :s, :v])
    @test BRM._rk_value_reads!(Set{Symbol}(), :(a = Normal.(m, t))) == Set([:m, :t])
    @test BRM._rk_value_reads!(Set{Symbol}(), :(z[1:n] .~ Normal.(0, s))) == Set([:s])

    # A definition's free reads: not its parameters, locals or sampled names.
    definition = :(block(a, b) = begin
        s ~ Exponential(rate)
        t = a + other
        x[1:b] ~ Normal(s, t)
        return x
    end)
    @test BRM._rk_free_reads!(Set{Symbol}(), definition) == Set([:rate, :other])
    @test BRM._rk_free_reads!(Set{Symbol}(), :(block(a; k=1) = begin
        s ~ Exponential(rate * k)
        return s + a
    end)) == Set([:rate])
    # An ordinary function has no caller scope to read a name from.
    @test isempty(BRM._rk_free_reads!(Set{Symbol}(), :(function helper(a)
        return a + rate
    end)))
    # A block's local is no read of the data field that shares its name.
    field_data = (; east=[0.1, 0.9, 0.7, 0.3], north=[0.2, 0.4, 0.8, 0.5],
        time=[0.0, 0.3, 0.3, 1.7], y=[0.4, -0.3, 0.2, 0.1],
        xy=[0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5], triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5],
        sigma=1.7)
    local_named = @brm field_data begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        y ~ Normal(field, 0.2)
    end
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(local_named))
    @test any(d -> :sigma in BRM._rk_source_outputs!(Set{Symbol}(),
        BRM._rk_definition_parts(d).statements), emitted.defs)
    @test !any(s -> Meta.isexpr(s, :(=)) && first(s.args) === :sigma, emitted.main.args)
end

@stestset "a gmrf prior's data scalar replays as a block prior's does" begin
    build = @brm begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(scale)
        y ~ Normal(field, 0.2)
    end
    train = (; east=[0.1, 0.9, 0.7, 0.3], north=[0.2, 0.4, 0.8, 0.5],
        time=[0.0, 0.3, 0.3, 1.7], y=[0.4, -0.3, 0.2, 0.1],
        xy=[0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5], triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5],
        scale=1.7)
    omitted = (; (k => v for (k, v) in pairs(train) if k !== :scale)...)
    changed = merge(train, (; scale=9.0))
    rk = RKBRMI(build(train))
    u = point(rk)
    @test :scale ∉ keys(rk_bound_data(rk))
    # A design may repeat the value or omit it.
    @test density(reprocess(rk, train), u) == density(rk, u)
    @test density(reprocess(rk, omitted), u) == density(rk, u)
    state = brm_fitted_state(rk)
    @test density(RKBRMI(build(omitted), state), u) == density(rk, u)
    binder = rk_design_binder(rk)
    @test isequal(binder(omitted), binder(train))
    # A design cannot change the prior, on any replay route.
    @test_throws "changes the fitted RK program" reprocess(rk, changed)
    @test_throws "changes the fitted RK program" RKBRMI(build(changed), state)
    @test_throws "changes the fitted RK program" binder(changed)
    # Positive control: the rebuild does read the new value.
    rebuilt = reprocess(rk, changed; freeze_constants=false)
    @test density(rebuilt, u) ≈ density(RKBRMI(build(changed)), u) rtol=1e-13
    @test density(rebuilt, u) != density(rk, u)
end

@stestset "data-free replay preserves scalar inputs of graph constants" begin
    # Runtime source builders and macro builders must freeze the same values.
    source = quote
        anchor_fit = constant(anchor)
        offset = abs(anchor_fit)
        shifted = x - offset
        sigma ~ Exponential(1)
        mu ~ 1 + shifted
        y ~ Normal(mu, sigma)
    end
    macro_build = @brm begin
        anchor_fit = constant(anchor)
        offset = abs(anchor_fit)
        shifted = x - offset
        sigma ~ Exponential(1)
        mu ~ 1 + shifted
        y ~ Normal(mu, sigma)
    end
    train = (; x=[-2.0, 1.0, 4.0], y=[0.2, 0.7, -0.1], anchor=-0.8)
    design = (; x=[2.0, 5.0], y=[0.3, -0.4])
    for build in (brm(source; mod=FM), macro_build)
        rk = RKBRMI(build(train))
        state = brm_fitted_state(rk)
        free = check_data_free(build, state, rk, design)
        @test graph_value(free, :shifted, point(free)) == design.x .- abs(train.anchor)
        conflicting = merge(design, (; anchor=9.0))
        @test isequal(rk_bound_data(RKBRMI(build(conflicting), state)), rk_bound_data(free))
        check_data_free(build, brm_fitted_state(free), rk, design)
        binder = rk_design_binder(free)
        trained = rk_design_binder(rk)
        for next in (design, (; x=[7.0], y=[0.6]), conflicting)
            check_inputs(binder(next), trained(next), state, rk)
        end
    end
end

@stestset "scalar replay re-exports mixed constants and resampled groups" begin
    build = brm(quote
        anchor_fit = constant(mean(reference) + anchor)
        shifted = x - abs(anchor_fit)
        sigma ~ Exponential(prior_scale)
        mu ~ 1 + shifted + (1 | subject)
        y ~ Normal(mu, sigma)
    end; mod=FM)
    train = (; x=[-2.0, 1.0, 4.0], y=[0.2, 0.7, -0.1], subject=[1, 1, 2],
        reference=[-3.0, 4.0, 5.0, 2.0], anchor=-0.8, prior_scale=1.7)
    rk = RKBRMI(build(train))
    state = brm_fitted_state(rk)
    @test state.scalars == Dict(:anchor => train.anchor, :prior_scale => train.prior_scale)
    @test state.constants[:anchor_fit] == mean(train.reference) + train.anchor
    for (subjects, groups) in (([1, 2], ()), ([101, 102], (:subject,)))
        design = (; x=[2.0, 5.0], y=[0.3, -0.4], subject=subjects)
        free = check_data_free(build, state, rk, design; resample_groups=groups)
        exported = brm_fitted_state(free)
        @test exported.scalars == state.scalars
        @test exported.columns == state.columns
        @test exported.constants == state.constants
        io = IOBuffer()
        serialize(io, exported)
        again = RKBRMI(build(design), deserialize(seekstart(io)))
        # Re-export freezes newly fitted group levels too: its raw group-fit
        # input is now absent, while every value that reads the levels agrees.
        check_inputs(rk_bound_data(again), rk_bound_data(free), exported, free)
        @test BRM._rk_layout_coordinate_names(again) == exported.coordinates
        @test density(again, point(again)) == density(free, point(free))
        @test graph_value(again, :shifted, point(again)) ==
            design.x .- abs(mean(train.reference) + train.anchor)
    end
end

@stestset "a design binds from the fitted state alone" begin
    for build in SUPPORTED
        rk = RKBRMI(build(TRAIN))
        state = brm_fitted_state(rk)
        check_data_free(build, state, rk, rows(TRAIN, 3:9))
        check_data_free(build, state, rk, rows(TRAIN, [5]))
        free = check_data_free(build, state, rk, without(rows(TRAIN, 2:6), :y))
        # A data-free backend replays further designs from its state.
        again = reprocess(free, rows(TRAIN, 10:14))
        check_inputs(rk_bound_data(again), rk_bound_data(reprocess(rk, rows(TRAIN, 10:14))),
            state, rk)
        # The state serialized and read back binds the same design.
        io = IOBuffer()
        serialize(io, state)
        back = deserialize(seekstart(io))
        @test isequal(rk_bound_data(RKBRMI(build(rows(TRAIN, 3:9)), back)),
            rk_bound_data(RKBRMI(build(rows(TRAIN, 3:9)), state)))
    end
end

@stestset "new subjects bind from the fitted state alone" begin
    # A resampled grouping refits its levels from the design; every other
    # fitted value stays frozen.
    rk = RKBRMI(FM.grouped(TRAIN))
    state = brm_fitted_state(rk)
    newcomers = merge(rows(TRAIN, 1:8), (; subject=repeat([101, 102], inner=4)))
    free = check_data_free(FM.grouped, state, rk, newcomers; resample_groups=[:subject])
    @test rk_bound_data(free).subject_fit_levels == [101, 102]
    @test rk_bound_data(free).x_fit_mean == state.fitted.values.x_fit_mean
end

@stestset "a value RK preparation folds is frozen too" begin
    # A completed covariate's anchors are read only by the parameter-dependent
    # completion, so RKPPL folds them at preparation, not at binding. The
    # design keeps every missing entry, as on `reprocess`.
    rk = RKBRMI(FM.completed(TRAIN))
    state = brm_fitted_state(rk)
    @test !isempty(state.fitted)
    check_data_free(FM.completed, state, rk, rows(TRAIN, [1, 5, 10, 15, 20, 21]))
end

@stestset "a data-free binder binds what the training binder binds" begin
    rk = RKBRMI(FM.grouped(TRAIN))
    state = brm_fitted_state(rk)
    free = RKBRMI(FM.grouped(rows(TRAIN, 1:4)), state)
    for groups in ((), (:subject,))
        binder, trained = rk_design_binder(free; resample_groups=groups),
            rk_design_binder(rk; resample_groups=groups)
        @test !any(name -> name in state.fit_inputs, keys(binder.fitted))
        # A resampled grouping's levels are the design's, never frozen.
        @test (:subject in groups) != haskey(binder.fitted, :subject_fit_levels)
        # The second and third designs are of the first one's kind: bound into
        # its lowering without planning.
        for design in (rows(TRAIN, 3:9), rows(TRAIN, 11:17), rows(TRAIN, 13:16))
            check_inputs(binder(design), trained(design), state, rk; resample_groups=groups)
        end
    end
end

@stestset "a design that omits a scalar the program reads stays on the cached lowering" begin
    # Snag `rk-design-binder-ddee0088`. A correlated block's scale prior reads
    # its argument as data, so the program binds the scalar. A design may leave
    # it out: every replay then binds the fit's value, on the training fit and
    # on a data-free replay alike.
    build = brm(quote
        anchor_fit = constant(anchor)
        shifted = w - anchor_fit
        sigma ~ Exponential(1)
        log_a ~ 1 + shifted + (1 | p | subject)
        log_b ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(prior_sd_scale)
        @plate for i in eachindex(log_a)
            mu = log_a[i] .+ log_b[i] .* t[i]
            y[i] ~ Normal(mu, sigma)
        end
    end; mod=FM)
    train = (; subject=[1, 2, 3], w=[1.0, 2.0, 3.0], t=[[0.5, 1.0], [0.25], [1.0, 2.0]],
        y=[[0.1, 0.2], [0.2], [0.3, 0.1]], anchor=-0.8, prior_sd_scale=1.7)
    rk = RKBRMI(build(train))
    state = brm_fitted_state(rk)
    @test haskey(rk_bound_data(rk), :prior_sd_scale)
    @test haskey(state.scalars, :prior_sd_scale)
    groups = [:subject]
    responses = (; subject=[101, 102], w=[1.5, 2.5], t=[[0.5], [1.0, 2.0]], y=[[0.3], [-0.4, 0.2]])
    more = (; subject=[103, 104, 105], w=[1.0, 2.0, 2.5], t=[[1.0], [0.5, 1.5], [2.0]],
        y=[[0.1], [0.2, 0.3], [0.4]])
    same(a, b) = Set(keys(a)) == Set(keys(b)) && all(k -> isequal(a[k], b[k]), keys(a))
    free = RKBRMI(build(responses), state; resample_groups=groups)
    for fit in (rk, free), (base, next) in ((responses, more),
            (without(responses, :y), without(more, :y)))
        planned(design) = rk_bound_data(reprocess(fit, design; resample_groups=groups))
        binder = rk_design_binder(fit; resample_groups=groups)
        for design in (base, next)
            inputs = binder(design)
            @test same(inputs, planned(design))
            @test inputs.prior_sd_scale == train.prior_sd_scale
        end
        @test !occursin("replayed in full", sprint(show, binder))
        class = only(values(binder.classes))
        @test class.lowering !== nothing
        @test last(only(s for s in class.sources if first(s) === :prior_sd_scale)) isa BRM._RKFixedInput

        # Equal by value to a column the design holds, it is still not that
        # column: a later design that changes the column leaves it alone.
        aliased = rk_design_binder(fit; resample_groups=groups)
        aliased(merge(base, (; anchor=train.prior_sd_scale)))
        changed = merge(next, (; anchor=9.0))
        @test aliased(changed).prior_sd_scale == train.prior_sd_scale
        @test same(aliased(changed), planned(changed))

        # A design that holds the scalar is another kind, read from the design.
        holding = rk_design_binder(fit; resample_groups=groups)
        holding(merge(base, (; prior_sd_scale=train.prior_sd_scale)))
        held = merge(next, (; prior_sd_scale=train.prior_sd_scale))
        @test same(holding(held), planned(held))
        @test !occursin("replayed in full", sprint(show, holding))
    end
end

@stestset "a thin-plate basis cannot leave its training values" begin
    # refused: `s(x)` keeps every training value of `x` as a knot, so no
    # data-free state can carry it (snag `no-data-free-exp-fec940cb`).
    @test_throws "thin-plate" brm_fitted_state(RKBRMI(FM.thin_plate(TRAIN)))
end

# Snag `rk-replay-read-s-1ec7a5c5`. A correlated block's scale prior, a cell
# latent's law and an assignment a prior reads keep a numeric scalar a read in
# the program, so the program is the same at any value and its identity check
# cannot see a design change it. `SCALAR_*` are one model's rows on each path.
const SCALAR_TRAIN = (; subject=[1, 2, 3], w=[1.0, 2.0, 3.0], t=[[0.5, 1.0], [0.25], [1.0, 2.0]],
    y=[[0.1, 0.2], [0.2], [0.3, 0.1]])
const SCALAR_DESIGN = (; subject=[1, 2, 3], w=[1.5, 2.5, 0.5], t=[[0.5], [1.0, 2.0], [0.75]],
    y=[[0.3], [-0.4, 0.2], [0.6]])
const SCALAR_LATER = (; subject=[3, 2, 1], w=[0.5, 1.0, 2.0], t=[[1.5], [0.5], [2.5, 0.25]],
    y=[[0.2], [0.1], [-0.3, 0.4]])
const SCALAR_NEW = merge(SCALAR_DESIGN, (; subject=[101, 102, 103]))
scalar_equal(a, b) = Set(keys(a)) == Set(keys(b)) && all(k -> isequal(a[k], b[k]), keys(a))

const PRIOR_READS = (
    "a correlated block's scale" => brm(quote
        sigma ~ Exponential(1)
        log_a ~ 1 + w + (1 | p | subject)
        log_b ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(s)
        @plate for i in eachindex(log_a)
            mu = log_a[i] .+ log_b[i] .* t[i]
            y[i] ~ Normal(mu, sigma)
        end
    end; mod=FM),
    "one block's scale" => brm(quote
        sigma ~ Exponential(1)
        log_a ~ 1 + w + (1 | subject)
        sd(log_a, subject) ~ Exponential(s)
        @plate for i in eachindex(log_a)
            mu = log_a[i] .+ t[i]
            y[i] ~ Normal(mu, sigma)
        end
    end; mod=FM),
    "through an assignment" => brm(quote
        sigma ~ Exponential(1)
        doubled = 2 * s
        log_a ~ 1 + w + (1 | p | subject)
        log_b ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(doubled)
        @plate for i in eachindex(log_a)
            mu = log_a[i] .+ log_b[i] .* t[i]
            y[i] ~ Normal(mu, sigma)
        end
    end; mod=FM),
    "read by a cell as well" => brm(quote
        sigma ~ Exponential(1)
        log_a ~ 1 + w + (1 | p | subject)
        log_b ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(s)
        @plate for i in eachindex(log_a)
            mu = s .* log_a[i] .+ log_b[i] .* t[i]
            y[i] ~ Normal(mu, sigma)
        end
    end; mod=FM),
    "a cell latent's law" => brm(quote
        sigma ~ Exponential(1)
        log_a ~ 1 + w + (1 | subject)
        @plate for i in eachindex(log_a)
            z ~ Normal(0, s)
            mu = log_a[i] .+ z .* t[i]
            y[i] ~ Normal(mu, sigma)
        end
    end; mod=FM),
)

@stestset "a replay refuses a changed scalar a prior reads by name" begin
    for (label, build) in PRIOR_READS
        @testset "$label" begin
            train = merge(SCALAR_TRAIN, (; s=1.7))
            rk = RKBRMI(build(train))
            state = brm_fitted_state(rk)
            @test state.scalars[:s] == train.s
            held(data, value=train.s) = merge(data, (; s=value))
            free = RKBRMI(build(SCALAR_DESIGN), state)
            frozen = reprocess(rk, SCALAR_DESIGN)
            u = point(frozen)
            @test density(free, u) == density(frozen, u)
            # The same value held by the design, omitted or not, is the same density.
            @test density(reprocess(rk, held(SCALAR_DESIGN)), u) == density(frozen, u)
            for fitted in (rk, free)
                for (data, groups) in ((SCALAR_DESIGN, ()), (SCALAR_NEW, (:subject,)))
                    label == "a cell latent's law" && !isempty(groups) && continue
                    @test reprocess(fitted, held(data); resample_groups=groups) isa RKBRMI
                    changed = held(data, 9.0)
                    @test_throws "changes the fitted RK program" reprocess(fitted, changed; resample_groups=groups)
                    @test_throws "prior input: s = 9.0" reprocess(fitted, changed; resample_groups=groups)
                end
                # A binder refuses it after the first design of the kind too.
                binder = rk_design_binder(fitted)
                @test scalar_equal(binder(held(SCALAR_DESIGN)),
                    rk_bound_data(reprocess(fitted, held(SCALAR_DESIGN))))
                @test scalar_equal(binder(held(SCALAR_LATER)),
                    rk_bound_data(reprocess(fitted, held(SCALAR_LATER))))
                @test_throws "changes the fitted RK program" binder(held(SCALAR_LATER, 9.0))
                @test !occursin("replayed in full", sprint(show, binder))
            end
            changed = held(SCALAR_DESIGN, 9.0)
            @test_throws "changes the fitted RK program" RKBRMI(build(changed), state)
            @test_throws "prior input: s = 9.0" RKBRMI(build(changed), state)
            # The value reaches the density, and the sanctioned rebuild takes it.
            fresh = reprocess(rk, changed; freeze_constants=false)
            @test density(fresh, u) == density(RKBRMI(build(changed)), u)
            @test density(fresh, u) != density(frozen, u)
        end
    end
end

@stestset "a scalar equal to the fitted value passes whatever its numeric type" begin
    # The same number is the same prior; the literal route folds both to Float64.
    build = last(first(PRIOR_READS))
    rk = RKBRMI(build(merge(SCALAR_TRAIN, (; s=2.0))))
    @test rk_bound_data(reprocess(rk, merge(SCALAR_DESIGN, (; s=2)))).s == 2
end

@stestset "a scalar a cell reads as data stays a design input" begin
    noise = brm(quote
        log_a ~ 1 + w + (1 | subject)
        @plate for i in eachindex(log_a)
            mu = log_a[i] .+ t[i]
            y[i] ~ Normal(mu, s)
        end
    end; mod=FM)
    dosed = brm(quote
        sigma ~ Exponential(1)
        log_a ~ 1 + w + (1 | subject)
        @plate for i in eachindex(log_a)
            mu = s .* exp.(log_a[i]) .* t[i]
            y[i] ~ Normal(mu, sigma)
        end
    end; mod=FM)
    for build in (noise, dosed)
        rk = RKBRMI(build(merge(SCALAR_TRAIN, (; s=1.7))))
        state = brm_fitted_state(rk)
        changed = merge(SCALAR_DESIGN, (; s=9.0))
        base = reprocess(rk, SCALAR_DESIGN)
        u = point(base)
        for fitted in (rk, RKBRMI(build(SCALAR_DESIGN), state))
            replay = reprocess(fitted, changed)
            @test density(replay, u) != density(base, u)
            # These rows hold every fitted level, so nothing else is frozen.
            @test density(replay, u) == density(RKBRMI(build(changed)), u)
            binder = rk_design_binder(fitted)
            for design in (merge(SCALAR_DESIGN, (; s=1.7)), merge(SCALAR_LATER, (; s=9.0)))
                @test scalar_equal(binder(design), rk_bound_data(reprocess(fitted, design)))
            end
            @test !occursin("replayed in full", sprint(show, binder))
        end
        @test density(RKBRMI(build(changed), state), u) == density(reprocess(rk, changed), u)
    end
end

@stestset "a binder refuses a changed scalar the program holds as a literal" begin
    # The program holds no input for it, so a later design's value had no way in
    # and was dropped, where `reprocess` refuses that same design.
    build = brm(quote
        sigma ~ Exponential(s)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end; mod=FM)
    train = (; x=[-2.0, 1.0, 4.0], y=[0.2, 0.7, -0.1], s=1.7)
    design = (; x=[2.0, 5.0], y=[0.3, -0.4], s=1.7)
    rk = RKBRMI(build(train))
    for fitted in (rk, RKBRMI(build(without(design, :s)), brm_fitted_state(rk)))
        binder = rk_design_binder(fitted)
        binder(design)
        later = (; x=[1.0], y=[0.5])
        @test scalar_equal(binder(merge(later, (; s=train.s))),
            rk_bound_data(reprocess(fitted, merge(later, (; s=train.s)))))
        @test_throws "changes the fitted RK program" binder(merge(later, (; s=9.0)))
    end
end

@stestset "a constant's input a prior reads through it keeps its fitted value" begin
    build = brm(quote
        sigma ~ Exponential(1)
        anchor_fit = constant(anchor)
        scale = abs(anchor_fit)
        log_a ~ 1 + w + (1 | p | subject)
        log_b ~ 1 + (1 | p | subject)
        sd(:, p) ~ Exponential(scale)
        @plate for i in eachindex(log_a)
            mu = log_a[i] .+ log_b[i] .* t[i]
            y[i] ~ Normal(mu, sigma)
        end
    end; mod=FM)
    rk = RKBRMI(build(merge(SCALAR_TRAIN, (; anchor=-0.8))))
    base = reprocess(rk, SCALAR_DESIGN)
    u = point(base)
    @test density(reprocess(rk, merge(SCALAR_DESIGN, (; anchor=9.0))), u) == density(base, u)
    binder = rk_design_binder(rk)
    for design in (merge(SCALAR_DESIGN, (; anchor=9.0)), merge(SCALAR_LATER, (; anchor=4.0)))
        @test scalar_equal(binder(design), rk_bound_data(reprocess(rk, design)))
    end
end

@stestset "a prior read is what a sampled parameter's law reads" begin
    defs = Expr[
        :(effects(X, n, loc, scale) = begin
            beta[1:n] .~ Normal.(loc, scale)
            return X * beta
        end),
        # `~` on a parameter observes what the caller passes.
        :(observed_part(v, m, noise) = begin
            v ~ Normal(m, noise)
        end),
        :(nested(g, scale) = begin
            inner ~ effects(g, 2, 0.0, scale)
            return inner
        end),
    ]
    main = quote
        a = 2 .* q
        b ~ effects(X, n, loc_in, a)
        c ~ restricted(Normal(0, rate), 0.0, Inf)
        e ~ observed_part(w, m_in, noise_in)
        f ~ nested(G, nested_scale)
        @plate for i in eachindex(t)
            z[i] .~ Normal.(0, cell_scale)
            mu = dose .* t[i]
            out[i] = mu
        end
        @plate for i in eachindex(y)
            y[i] .~ Normal.(out[i], known_sd)
        end
    end
    reads = BRM._rk_prior_reads(BRM._RKEmittedProgram(defs, main), (:y,))
    @test issubset((:loc_in, :a, :q, :rate, :cell_scale, :nested_scale), reads)
    @test isdisjoint(reads, (:X, :n, :G, :w, :m_in, :noise_in, :dose, :t, :out, :known_sd, :y))
end
