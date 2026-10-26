# test/fitted_constants.jl — `name = constant(expr)`: a scalar calibrated on
# the data when the model is built and frozen at that value on replay (snag
# `bruno-fit-time-d-dba90310`, user design decision `00kjor3`), or an array of
# such numbers (snag `brm-constant-exp-86604df5`).
#
# A constant is evaluated once on whole columns and is then an ordinary data
# field: Stan data on SBBRMI, a bound value in the RKBRMI graph. Replay keeps
# the fitted value, so `reprocess`, `reprocess(...; resample_groups)` and
# `generative_plan(plan, new_df)` need no inputs for it, while
# `reprocess(...; freeze_constants=false)` evaluates it again on the new data.
#
# Run: julia --startup-file=no --project=test test/fitted_constants.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Serialization: serialize, deserialize
using Statistics: mean, std, var
using LinearAlgebra: cholesky, Symmetric
using LogExpFunctions: logistic
const stan_code = BRM.stan_code
const stan_data = BRM.stan_data

const PANEL = (;
    conc=[0.4, 1.1, 2.3, 0.7, 1.9, 3.1, 0.5, 1.4, 2.8, 0.9, 1.6, 2.2],
    x=[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, -0.9, -0.3, 0.1, 0.5, 0.8, 1.2],
    subject=repeat(1:4, inner=3),
    y=[0.3, 0.9, 1.6, 0.4, 1.2, 2.0, 0.1, 0.8, 1.7, 0.6, 1.1, 1.5],
)
# A Julia value a constant reads from the builder's scope.
const OFFSET = 0.5

const LOC = mean(log.(PANEL.conc))
const SCALE = std(log.(PANEL.conc))
const V = var(PANEL.y)
const CAP = maximum(log.(PANEL.conc)) + OFFSET

const CALIBRATED = @brm begin
    ec50_loc = constant(mean(log.(conc)))
    ec50_scale = constant(std(log.(conc)))
    V = constant(var(y))
    log_ec50 ~ Normal(ec50_loc, ec50_scale)
    mu ~ 1 + x + (1 | p | subject)
    r2 ~ Beta(1.0, 1.0)
    sigma = sqrt((1 - r2) * V)
    m = mu + log_ec50
    y ~ Normal(m, sigma)
end
# The same model with the three scalars supplied as data fields.
const SUPPLIED = @brm begin
    log_ec50 ~ Normal(ec50_loc, ec50_scale)
    mu ~ 1 + x + (1 | p | subject)
    r2 ~ Beta(1.0, 1.0)
    sigma = sqrt((1 - r2) * V)
    m = mu + log_ec50
    y ~ Normal(m, sigma)
end
supplied(data) = merge(data, (; ec50_loc=LOC, ec50_scale=SCALE, V))

# Replay frames: the inputs `conc` absent, and a one-row grid on which the
# constants are undefined (`std` of one value is NaN).
const NO_CONC = (; x=PANEL.x[1:6], subject=PANEL.subject[1:6], y=PANEL.y[1:6])
const ONE_ROW = (; conc=[2.0], x=[0.5], subject=[11], y=[0.2])
const ONE_ROW_NO_CONC = (; x=[0.5], subject=[11], y=[0.2])

constants(sb) = Dict(k => stan_data(sb)[k] for k in (:ec50_loc, :ec50_scale, :V))
const FITTED = Dict(:ec50_loc => LOC, :ec50_scale => SCALE, :V => V)

@stestset "a constant is the scalar data field it computes" begin
    saved = deepcopy(PANEL)
    brmi = CALIBRATED(PANEL)
    @test occursin("ec50_loc = constant(mean(log.(conc)))", sprint(show, brmi))
    sb = SBBRMI(brmi; mod=@__MODULE__)
    reference = SBBRMI(SUPPLIED(supplied(PANEL)); mod=@__MODULE__)
    @test stan_code(sb) == stan_code(reference)
    @test constants(sb) == FITTED
    @test Dict(k => e.kind for (k, e) in sb.preproc if e.kind === :constant) ==
        Dict(:ec50_loc => :constant, :ec50_scale => :constant, :V => :constant)
    @test StanBlocks.stan.compiles(sb.model)
    @test isequal(PANEL, saved)
end

@stestset "RKBRMI holds constants in prior arguments and bounds, matching Stan" begin
    builder = @brm begin
        loc = constant(mean(log.(conc)))
        scale = constant(std(log.(conc)))
        V = constant(var(y))
        cap = constant(maximum(log.(conc)) + OFFSET)
        log_ec50 ~ Normal(loc, scale; upper=cap)
        r2 ~ Beta(1.0, 1.0)
        sigma = sqrt((1 - r2) * V)
        y ~ Normal(log_ec50, sigma)
    end
    brmi = builder(PANEL)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    @test length(names) == 2
    ie, ir = coordinate(names, :log_ec50), coordinate(names, :r2)
    oracle(u) = begin
        log_ec50, r2 = CAP - exp(u[ie]), logistic(u[ir])
        sigma = sqrt((1 - r2) * V)
        logpdf(Normal(LOC, SCALE), log_ec50) + u[ie] +
            logpdf(Beta(1.0, 1.0), r2) + log(r2) + log1p(-r2) +
            sum(logpdf.(Normal(log_ec50, sigma), PANEL.y))
    end
    stan = consumer_stan(brmi, "fitted-constants-bounds")
    mapping = [names[ie] => "log_ec50", names[ir] => "r2"]
    for u in (zeros(2), [-0.4, 0.7], [0.3, -1.2])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    # Supplying the same scalars as data fields gives the same graph values.
    supplied_backend, supplied_problem = consumer_problem((@brm begin
        log_ec50 ~ Normal(loc, scale; upper=cap)
        r2 ~ Beta(1.0, 1.0)
        sigma = sqrt((1 - r2) * V)
        y ~ Normal(log_ec50, sigma)
    end)(merge(PANEL, (; loc=LOC, scale=SCALE, V, cap=CAP))))
    @test coordinate_names(supplied_backend.model.layout) == names
    for u in (zeros(2), [-0.4, 0.7])
        @test LogDensityProblems.logdensity(supplied_problem, u) ≈
            LogDensityProblems.logdensity(problem, u) atol=1e-12
    end
end

@stestset "a constant scales a population-effect prior on both backends" begin
    calibrated = @brm begin
        slope_scale = constant(2 * std(y))
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, slope_scale)
        y ~ Normal(mu, sigma)
    end
    supplied_scale = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, slope_scale)
        y ~ Normal(mu, sigma)
    end
    with_scale = merge(PANEL, (; slope_scale=2 * std(PANEL.y)))
    sb = SBBRMI(calibrated(PANEL); mod=@__MODULE__, total_groups=())
    @test stan_code(sb) == stan_code(SBBRMI(supplied_scale(with_scale); mod=@__MODULE__, total_groups=()))
    @test stan_data(sb)[:slope_scale] == 2 * std(PANEL.y)
    @test StanBlocks.stan.compiles(sb.model)
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = rk_logdensity_problem(RKBRMI(calibrated(PANEL)); ad_backend)
    rk_supplied = rk_logdensity_problem(RKBRMI(supplied_scale(with_scale)); ad_backend)
    literal_scale = eval(:(@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, $(2 * std(PANEL.y)))
        y ~ Normal(mu, sigma)
    end))
    rk_literal = rk_logdensity_problem(RKBRMI(literal_scale(PANEL)); ad_backend)
    for u in (zeros(LogDensityProblems.dimension(rk)), fill(0.3, LogDensityProblems.dimension(rk)))
        @test LogDensityProblems.logdensity(rk, u) == LogDensityProblems.logdensity(rk_supplied, u)
        @test LogDensityProblems.logdensity(rk, u) ≈ LogDensityProblems.logdensity(rk_literal, u) atol=1e-12
    end
    # A sampled scale is still refused.
    @test_throws "must be a numeric constant or a scalar data field" RKBRMI(@brm PANEL begin
        slope_scale ~ Exponential(1)
        sigma ~ Exponential(1)
        mu ~ 1 + x
        effect(mu, x) ~ Normal(0, slope_scale)
        y ~ Normal(mu, sigma)
    end)
end

@stestset "replay keeps fitted constants without their inputs" begin
    sb = SBBRMI(CALIBRATED(PANEL); mod=@__MODULE__, total_groups=())
    frozen = reprocess(sb, NO_CONC)
    @test constants(frozen) == FITTED
    @test stan_code(frozen) == stan_code(sb)
    # A one-row grid would make `std` NaN; the fitted value is kept instead.
    for frame in (ONE_ROW, ONE_ROW_NO_CONC)
        resampled = reprocess(sb, frame; resample_groups=[:subject])
        @test constants(resampled) == FITTED
        @test count(e -> e.kind === :constant, values(resampled.preproc)) == 3
    end
end

@stestset "fresh-constant replay evaluates the constant on the new data" begin
    sb = SBBRMI(CALIBRATED(PANEL); mod=@__MODULE__, total_groups=())
    conc = PANEL.conc[1:6] .* 10
    fresh = reprocess(sb, merge(NO_CONC, (; conc)); freeze_constants=false)
    @test constants(fresh) == Dict(:ec50_loc => mean(log.(conc)),
        :ec50_scale => std(log.(conc)), :V => var(NO_CONC.y))
    @test stan_code(fresh) == stan_code(sb)
    err = try
        reprocess(sb, NO_CONC; freeze_constants=false); nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("reads `conc`", err.msg)
    err = try
        reprocess(sb, ONE_ROW; resample_groups=[:subject], freeze_constants=false); nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("one finite real number or an array of them", err.msg)
    # A scalar the data supplies directly is taken from the new data as well.
    plain = SBBRMI(SUPPLIED(supplied(PANEL)); mod=@__MODULE__, total_groups=())
    replay = merge(NO_CONC, (; ec50_loc=9.0, ec50_scale=1.0, V=2.0))
    @test constants(reprocess(plain, replay; freeze_constants=false)) ==
        Dict(:ec50_loc => 9.0, :ec50_scale => 1.0, :V => 2.0)
    @test constants(reprocess(plain, replay)) == FITTED
end

@stestset "a plan rebuilt for new groups keeps fitted constants" begin
    for total_groups in (:auto, ())
        plan = generative_plan(CALIBRATED, PANEL; mod=@__MODULE__, total_groups)
        @test (total_groups === :auto) == !isempty(total_effect_blocks(plan))
        rebuilt = generative_plan(plan, ONE_ROW_NO_CONC)
        @test Dict(k => rebuilt.data[k] for k in keys(FITTED)) == FITTED
    end
end

@stestset "the descriptor reports a constant as a :constant input" begin
    descriptor = brm_descriptor(CALIBRATED, PANEL; mod=@__MODULE__)
    for name in keys(FITTED)
        input = only(i for i in descriptor.inputs if i.name === name)
        @test input.transform === :constant
        @test isnothing(input.column)
    end
    # Replay keeps the fitted value, so `conc` is not part of the replay schema.
    @test :conc ∉ brm_columns(descriptor)
    @test :y in brm_columns(descriptor)
    # Each constant is a covered component naming its expression and value.
    description = brm_description(descriptor)
    calls = filter(c -> c.callable === constant, collect(brm_description_components(description)))
    @test Set(c.provenance.owner for c in calls) == Set(keys(FITTED))
    for c in calls
        @test only(x for x in description.coverage if x.id == c.id).status === :covered
    end
    markdown = brm_description_markdown(description)
    @test occursin("operatorname{constant}", markdown)
    @test occursin("as `mean(log.(conc))`. Its fitted value is $LOC", markdown)
    supplied_description = brm_description(brm_descriptor(SUPPLIED, supplied(PANEL); mod=@__MODULE__))
    @test description.complete == supplied_description.complete
end

@stestset "a constant reads a local of the builder's scope and serves a plate" begin
    panel = (; subject_id=1:4, cobs=[PANEL.conc[3k-2:3k] for k in 1:4],
        yobs=[PANEL.y[3k-2:3k] for k in 1:4], wt=[60.0, 70.0, 80.0, 75.0])
    builder(shift) = @brm begin
        ref = constant(maximum(reduce(vcat, cobs)) + shift)
        sigma ~ Exponential(1)
        slope ~ 1 + wt + (1 | p | subject_id)
        @plate for i in eachindex(slope)
            pred = slope[i] .* cobs[i] ./ ref
            yobs[i] ~ normal(pred, sigma)
        end
    end
    sb = SBBRMI(builder(OFFSET)(panel); mod=@__MODULE__)
    @test stan_data(sb)[:ref] == maximum(PANEL.conc) + OFFSET
    @test stan_data(SBBRMI(builder(2.0)(panel); mod=@__MODULE__))[:ref] ==
        maximum(PANEL.conc) + 2.0
    @test StanBlocks.stan.compiles(sb.model)
    @test RKBRMI(builder(OFFSET)(panel)) isa RKBRMI
end

# ---- vector constants (snag `brm-constant-exp-86604df5`) -------------------
#
# A constant may be a vector of finite reals, its length fixed when it is
# evaluated: a data vector on SBBRMI, a bound vector on RKBRMI, recorded and
# frozen on every replay path exactly like a scalar. The model below reads a
# grid calibrated on the training times in every subject's plate cell.

# Five offsets doubling up to the largest training time.
time_grid(t) = maximum(t) .* (2.0 .^ (-4:0))

const CELLS = (; subject_id=1:4,
    tobs=[[0.5, 1.0, 2.0], [0.3, 1.5, 2.5], [1.0, 2.0, 3.0], [0.2, 0.8, 1.6]],
    yobs=[[0.4, 0.7, 1.5], [0.2, 1.1, 1.9], [0.9, 1.4, 2.4], [0.1, 0.6, 1.2]],
    wt=[0.6, 0.7, 0.8, 0.75])
const GRID = time_grid(reduce(vcat, CELLS.tobs))

const GRIDDED = @brm begin
    grid = constant(time_grid(reduce(vcat, tobs)))
    sigma ~ Exponential(1)
    rate ~ 1 + wt + (1 | p | subject_id)
    @plate for i in eachindex(rate)
        pred = rate[i] .* tobs[i] ./ sum(grid)
        yobs[i] ~ normal(pred, sigma)
    end
end
# The same model with the grid supplied as a data field.
const GRID_SUPPLIED = @brm begin
    sigma ~ Exponential(1)
    rate ~ 1 + wt + (1 | p | subject_id)
    @plate for i in eachindex(rate)
        pred = rate[i] .* tobs[i] ./ sum(grid)
        yobs[i] ~ normal(pred, sigma)
    end
end
with_grid(data, grid=GRID) = merge(data, (; grid))

# A replay design: two of the fitted subjects with later times and no
# responses, on which `time_grid` would give a different grid.
const LATER = (; subject_id=[2, 3], tobs=[[4.0, 8.0], [6.0]], wt=[0.7, 0.8])
const LATER_GRID = time_grid(reduce(vcat, LATER.tobs))
# The same design with responses: a density, and a fresh-constant replay
# (which may not omit responses), read them.
const LATER_OBS = merge(LATER, (; yobs=[[1.0, 2.0], [1.5]]))
# New subjects with responses, for a replay that resamples the grouping (an
# SBBRMI forecast cannot resample yet, todo `1fnt1zu`).
const NEW_SUBJECTS = (; subject_id=[11, 12], tobs=[[4.0, 8.0], [6.0]],
    yobs=[[1.0, 2.0], [1.5]], wt=[0.5, 0.9])

rk_value(problem, u) = LogDensityProblems.logdensity(problem, u)

@stestset "a vector constant is the data vector it computes" begin
    saved = deepcopy(CELLS)
    sb = SBBRMI(GRIDDED(CELLS); mod=@__MODULE__)
    reference = SBBRMI(GRID_SUPPLIED(with_grid(CELLS)); mod=@__MODULE__)
    @test stan_code(sb) == stan_code(reference)
    @test stan_data(sb)[:grid] == GRID
    @test stan_data(sb)[:grid] isa Vector{Float64}
    @test sb.preproc[:grid].kind === :constant
    @test StanBlocks.stan.compiles(sb.model)
    # The compiled programs give the same density at the same coordinates.
    names, densities = stan_density_points(GRIDDED(CELLS), "vector-constant")
    supplied_names, supplied_densities = stan_density_points(
        GRID_SUPPLIED(with_grid(CELLS)), "vector-constant-supplied")
    @test names == supplied_names
    @test densities == supplied_densities
    # RKBRMI binds the grid: the graph is the supplied model's.
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = RKBRMI(GRIDDED(CELLS))
    @test rk_bound_data(rk).grid == GRID
    problem = rk_logdensity_problem(rk; ad_backend)
    supplied = rk_logdensity_problem(RKBRMI(GRID_SUPPLIED(with_grid(CELLS))); ad_backend)
    n = LogDensityProblems.dimension(problem)
    @test n == LogDensityProblems.dimension(supplied)
    for u in (zeros(n), fill(0.3, n), collect(range(-0.4, 0.4; length=n)))
        @test rk_value(problem, u) == rk_value(supplied, u)
    end
    @test isequal(CELLS, saved)
end

@stestset "a vector constant's value never aliases what its expression returned" begin
    # `constant(x)` evaluates to the data column itself; the fitted value is a
    # copy, so changing the container afterwards leaves the model unchanged.
    data = (; x=[1.0, 2.0, 3.0], y=[0.1, 0.4, 0.2])
    builder = @brm begin
        xs = constant(x)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = b * xs
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(builder(data); mod=@__MODULE__)
    @test stan_data(sb)[:xs] == data.x
    @test stan_data(sb)[:xs] !== data.x
    data.x[1] = 10.0
    @test stan_data(sb)[:xs] == [1.0, 2.0, 3.0]
    @test sb.preproc[:xs].const_.value == [1.0, 2.0, 3.0]
    # The admitted value: a fresh `Array` of the elements' promoted type, so
    # integers stay integers (as in a supplied data field) and a range, a view
    # or a loosely typed array is materialised; its shape is the value's.
    @test BRM._brm_constant_value(1:4) == [1, 2, 3, 4]
    @test BRM._brm_constant_value(1:4) isa Vector{Int}
    @test BRM._brm_constant_value(Any[1.0, 2]) isa Vector{Float64}
    @test BRM._brm_constant_value(Union{Missing,Float64}[1.0, 2.0]) isa Vector{Float64}
    @test BRM._brm_constant_value(Float64[]) == Float64[]
    v = [0.5, 1.5]
    @test BRM._brm_constant_value(v) == v && BRM._brm_constant_value(v) !== v
    @test BRM._brm_constant_value(2.5) === 2.5
    m = [1.0 0.5; 0.5 1.0]
    @test BRM._brm_constant_value(m) == m && BRM._brm_constant_value(m) !== m
    @test BRM._brm_constant_value(m') isa Matrix{Float64}
    @test BRM._brm_constant_value(view(m, :, 1)) isa Vector{Float64}
    @test BRM._brm_constant_value(reshape(1.0:8.0, 2, 2, 2)) isa Array{Float64,3}
    @test BRM._brm_constant_value(Any[1 2.0; 3 4]) isa Matrix{Float64}
    for refused in ([1.0, NaN], [true, false], [1.0, missing], [1.0 Inf], [true false],
                    ["a" "b"], "a", true, -Inf)
        @test BRM._brm_constant_value(refused) === nothing
    end
end

@stestset "replay keeps a fitted vector constant" begin
    # Conventional random effects: `resample_groups` does not recover exact
    # totals (that is `generative_plan`, below).
    sb = SBBRMI(GRIDDED(CELLS); mod=@__MODULE__, total_groups=())
    reference = SBBRMI(GRID_SUPPLIED(with_grid(CELLS)); mod=@__MODULE__, total_groups=())
    for (design, kw) in ((LATER, (;)), (NEW_SUBJECTS, (; resample_groups=[:subject_id])))
        frozen = reprocess(sb, design; kw...)
        @test stan_data(frozen)[:grid] == GRID
        @test frozen.preproc[:grid] === sb.preproc[:grid]
        # The replay is the data-field twin's replay of the design carrying
        # the fitted grid: same program, same data.
        twin = reprocess(reference, with_grid(design); kw...)
        @test stan_code(frozen) == stan_code(twin)
        @test stan_data(frozen) == stan_data(twin)
    end
    # The fitted value is not shared with the replay's data.
    @test stan_data(reprocess(sb, LATER))[:grid] !== sb.preproc[:grid].const_.value
    fresh = reprocess(sb, LATER; freeze_constants=false)
    @test stan_data(fresh)[:grid] == LATER_GRID
    @test fresh.preproc[:grid].const_.value == LATER_GRID
    # A rebuilt plan for new groups keeps the fitted grid.
    plan = generative_plan(GRIDDED, CELLS; mod=@__MODULE__)
    @test plan.data[:grid] == GRID
    @test generative_plan(plan, NEW_SUBJECTS).data[:grid] == GRID
end

@stestset "RKBRMI replays, binds designs and exports a fitted vector constant" begin
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = RKBRMI(GRIDDED(CELLS))
    supplied = RKBRMI(GRID_SUPPLIED(with_grid(CELLS)))
    for (design, kw) in ((LATER, (;)), (NEW_SUBJECTS, (; resample_groups=[:subject_id])))
        replay = reprocess(rk, design; kw...)
        @test rk_bound_data(replay).grid == GRID
        # The replay is the supplied model's replay of the same design.
        twin = reprocess(supplied, with_grid(design); kw...)
        @test rk_bound_data(twin).grid == GRID
        # A design binder binds what the planned replay binds.
        binder = rk_design_binder(rk; kw...)
        @test binder(design).grid == GRID
        @test binder(design).tobs == rk_bound_data(replay).tobs
    end
    @test :grid in keys(rk_design_binder(rk).fitted)
    # `freeze_constants=false` evaluates the grid on the design.
    @test rk_bound_data(reprocess(rk, LATER_OBS; freeze_constants=false)).grid == LATER_GRID
    # The data-free fitted state carries the grid, and a design bound from the
    # state alone gives the training fit's replay of it.
    state = brm_fitted_state(rk)
    @test state.constants[:grid] == GRID
    restored = deserialize(IOBuffer(sprint(serialize, state)))
    @test restored.constants[:grid] == GRID
    @test rk_bound_data(RKBRMI(GRIDDED(LATER), restored)).grid == GRID
    @test rk_bound_data(RKBRMI(GRIDDED(NEW_SUBJECTS), restored;
        resample_groups=[:subject_id])).grid == GRID
    # The density of the fitted subjects on later times (new subjects' effects
    # are supplied as values, not sampled, so they have no such density).
    let design = LATER_OBS
        bound = RKBRMI(GRIDDED(design), restored)
        @test rk_bound_data(bound).grid == GRID
        replay = reprocess(rk, design)
        problem = rk_logdensity_problem(bound; ad_backend)
        reference = rk_logdensity_problem(replay; ad_backend)
        n = LogDensityProblems.dimension(problem)
        @test n == LogDensityProblems.dimension(reference)
        for u in (zeros(n), fill(0.2, n))
            @test rk_value(problem, u) == rk_value(reference, u)
        end
    end
end

@stestset "the descriptor and the description report a vector constant" begin
    descriptor = brm_descriptor(GRIDDED, CELLS; mod=@__MODULE__)
    input = only(i for i in descriptor.inputs if i.name === :grid)
    @test input.transform === :constant
    @test isnothing(input.column)
    @test :grid ∉ brm_columns(descriptor)
    description = brm_description(descriptor)
    call = only(c for c in brm_description_components(description) if c.callable === constant)
    @test only(x for x in description.coverage if x.id == call.id).status === :covered
    markdown = brm_description_markdown(description)
    @test occursin("Its fitted value is ($(join(GRID, ", ")))", markdown)
end

@stestset "a vector constant is a prior's data vector" begin
    # A declared vector parameter and per-coefficient effect priors read a
    # vector constant as they read the data vector it computes.
    data = (; y=[0.4, 1.3, -0.2, 2.1, 0.9, 1.7], g=["a", "b", "c", "d", "a", "c"])
    centres = [mean(data.y[data.g .== l]) for l in ("a", "b", "c", "d")]
    calibrated = @brm begin
        locs = constant([mean(y[g .== l]) for l in ("a", "b", "c", "d")])
        phi[1:4] .~ Normal.(locs, 1.0)
        sigma ~ Exponential(1)
        w ~ 0 + g
        effect(w, g) .~ Normal.(locs, 2.0)
        mu = phi[1] + w
        y ~ Normal(mu, sigma)
    end
    supplied_locs = @brm begin
        phi[1:4] .~ Normal.(locs, 1.0)
        sigma ~ Exponential(1)
        w ~ 0 + g
        effect(w, g) .~ Normal.(locs, 2.0)
        mu = phi[1] + w
        y ~ Normal(mu, sigma)
    end
    with_locs = merge(data, (; locs=centres))
    sb = SBBRMI(calibrated(data); mod=@__MODULE__, total_groups=())
    @test stan_code(sb) == stan_code(SBBRMI(supplied_locs(with_locs); mod=@__MODULE__, total_groups=()))
    @test stan_data(sb)[:locs] == centres
    @test StanBlocks.stan.compiles(sb.model)
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = rk_logdensity_problem(RKBRMI(calibrated(data)); ad_backend)
    rk_supplied = rk_logdensity_problem(RKBRMI(supplied_locs(with_locs)); ad_backend)
    n = LogDensityProblems.dimension(rk)
    for u in (zeros(n), fill(0.3, n))
        @test rk_value(rk, u) == rk_value(rk_supplied, u)
    end
end

# ---- array constants of any rank -------------------------------------------
#
# A constant's shape is not restricted: it is the data field it computes, so a
# matrix constant goes wherever a data matrix does. The model below correlates
# the levels of `g` by a known correlation calibrated on the training times.

# A correlation of the four levels that decays with level distance on the scale
# of the largest time (an exponential kernel, so positive definite).
level_cor(t) = [exp(-abs(i - j) / maximum(t)) for i in 1:4, j in 1:4]

const LEVELS = (; t=collect(range(0.5, 6.0; length=16)), g=repeat(1:4, 4),
    y=[0.3, -0.2, 0.8, 0.1, 0.5, 0.0, 1.1, 0.4, 0.9, 0.2, 1.4, 0.6, 1.2, 0.5, 1.6, 0.9])
const LEVEL_COR = level_cor(LEVELS.t)

const LEVEL_CORRELATED = @brm begin
    C = constant(level_cor(t))
    mu ~ 1 + t + (1 + t | gr(g; cor=C))
    y ~ Normal(mu, 1.0)
end
# The same model with the correlation supplied as a data field.
const LEVEL_COR_SUPPLIED = @brm begin
    mu ~ 1 + t + (1 + t | gr(g; cor=C))
    y ~ Normal(mu, 1.0)
end
with_cor(data, C=LEVEL_COR) = merge(data, (; C))
# A replay design on later times, on which `level_cor` gives another
# correlation.
const LEVELS_LATER = (; t=[8.0, 9.0, 10.0, 12.0], g=[1, 2, 3, 4], y=[1.0, 0.7, 1.9, 1.2])
const LATER_COR = level_cor(LEVELS_LATER.t)

@stestset "a matrix constant is the data matrix it computes" begin
    sb = SBBRMI(LEVEL_CORRELATED(LEVELS); mod=@__MODULE__)
    reference = SBBRMI(LEVEL_COR_SUPPLIED(with_cor(LEVELS)); mod=@__MODULE__)
    @test stan_code(sb) == stan_code(reference)
    @test stan_data(sb) == stan_data(reference)
    @test sb.preproc[:C].kind === :constant
    @test sb.preproc[:C].const_.value == LEVEL_COR
    @test sb.preproc[:C].const_.value isa Matrix{Float64}
    @test StanBlocks.stan.compiles(sb.model)
    names, densities = stan_density_points(LEVEL_CORRELATED(LEVELS), "matrix-constant")
    supplied_names, supplied_densities = stan_density_points(
        LEVEL_COR_SUPPLIED(with_cor(LEVELS)), "matrix-constant-supplied")
    @test names == supplied_names
    @test densities == supplied_densities
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    problem = rk_logdensity_problem(RKBRMI(LEVEL_CORRELATED(LEVELS)); ad_backend)
    supplied = rk_logdensity_problem(RKBRMI(LEVEL_COR_SUPPLIED(with_cor(LEVELS))); ad_backend)
    n = LogDensityProblems.dimension(problem)
    @test n == LogDensityProblems.dimension(supplied)
    for u in (zeros(n), fill(0.3, n), collect(range(-0.4, 0.4; length=n)))
        @test rk_value(problem, u) == rk_value(supplied, u)
    end
    # refused: a block has one correlation for every frequency, so
    # `multigroup(...)` is for a grouped `hsgp` only, whatever it wraps.
    blocked = @brm begin
        C = constant(level_cor(t))
        mu ~ 1 + t + (1 + t | gr(g; cor=multigroup(C, 1.0)))
        y ~ Normal(mu, 1.0)
    end
    @test_throws "is refused: `multigroup(...)`" SBBRMI(blocked(LEVELS); mod=@__MODULE__)
end

@stestset "replay keeps a fitted matrix constant" begin
    sb = SBBRMI(LEVEL_CORRELATED(LEVELS); mod=@__MODULE__, total_groups=())
    reference = SBBRMI(LEVEL_COR_SUPPLIED(with_cor(LEVELS)); mod=@__MODULE__, total_groups=())
    frozen = reprocess(sb, LEVELS_LATER)
    @test frozen.preproc[:C].const_.value == LEVEL_COR
    twin = reprocess(reference, with_cor(LEVELS_LATER))
    @test stan_code(frozen) == stan_code(twin)
    @test stan_data(frozen) == stan_data(twin)
    fresh = reprocess(sb, LEVELS_LATER; freeze_constants=false)
    @test fresh.preproc[:C].const_.value == LATER_COR
    @test stan_data(fresh)[:L_levelcor_C] ≈ Matrix(cholesky(Symmetric(LATER_COR)).L)
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = RKBRMI(LEVEL_CORRELATED(LEVELS))
    replay = rk_logdensity_problem(reprocess(rk, LEVELS_LATER); ad_backend)
    rk_twin = rk_logdensity_problem(
        reprocess(RKBRMI(LEVEL_COR_SUPPLIED(with_cor(LEVELS))), with_cor(LEVELS_LATER)); ad_backend)
    n = LogDensityProblems.dimension(replay)
    @test n == LogDensityProblems.dimension(rk_twin)
    for u in (zeros(n), fill(0.2, n))
        @test rk_value(replay, u) == rk_value(rk_twin, u)
    end
    # The data-free fitted state carries the matrix.
    state = deserialize(IOBuffer(sprint(serialize, brm_fitted_state(rk))))
    @test state.constants[:C] == LEVEL_COR
    bound = rk_logdensity_problem(RKBRMI(LEVEL_CORRELATED(LEVELS_LATER), state); ad_backend)
    for u in (zeros(n), fill(0.2, n))
        @test rk_value(bound, u) == rk_value(replay, u)
    end
end

@stestset "the description reports a matrix constant with its shape" begin
    markdown = brm_description_markdown(brm_description(
        brm_descriptor(LEVEL_CORRELATED, LEVELS; mod=@__MODULE__)))
    @test occursin("Its fitted value is (size = (4, 4), values = ($(join(LEVEL_COR, ", "))))", markdown)
end

@stestset "a constant of higher rank is the data field it computes" begin
    # The constant adds no rule of its own about shape: a rank-3 constant
    # resolves to the rank-3 data field it computes, as a supplied one would.
    data = (; x=[0.2, 0.5, 0.9, 1.4], y=[0.3, 0.6, 1.1, 1.5])
    A = reshape(collect(1.0:8.0), 2, 2, 2) ./ maximum(data.y)
    cubed = @brm begin
        A = constant(reshape(collect(1.0:8.0), 2, 2, 2) ./ maximum(y))
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = b * x + A[1, 2, 2]
        y ~ Normal(mu, sigma)
    end
    cubed_supplied = @brm begin
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = b * x + A[1, 2, 2]
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(cubed(data); mod=@__MODULE__)
    @test sb.preproc[:A].kind === :constant
    @test sb.preproc[:A].const_.value == A
    @test sb.preproc[:A].const_.value isa Array{Float64,3}
    supplied = SBBRMI(cubed_supplied(merge(data, (; A))); mod=@__MODULE__)
    # Capability gap, for a supplied rank-3 field and a constant alike:
    # StanBlocks has no Stan data type for a rank-3 array yet (StanBlocks snag
    # `stan-type-has-no-b2d832be`).
    transpiles(sb) = try
        stan_code(sb); true
    catch
        false
    end
    @test_broken transpiles(supplied)
    @test_broken transpiles(sb)
end

# A body built from a runtime string and `Core.eval`'d inside a function: the
# eval defines each constant's closure in a world newer than the function's,
# and the backends built in that same function evaluate it there (snag
# `declared-a-data-8e5f1f07`).
calibrate(conc) = (mean(log.(conc)), std(log.(conc)))
const RUNTIME_BODY = """
    ec50_loc = constant(calibrate(conc)[1])
    ec50_scale = constant(calibrate(conc)[2])
    V = constant(var(y))
    log_ec50 ~ Normal(ec50_loc, ec50_scale)
    mu ~ 1 + x + (1 | p | subject)
    r2 ~ Beta(1.0, 1.0)
    sigma = sqrt((1 - r2) * V)
    m = mu + log_ec50
    y ~ Normal(m, sigma)
"""
function build_from_runtime_body(body, data, replay)
    brmi = Core.eval(@__MODULE__, _brm(body; df=data))
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    builder = Core.eval(@__MODULE__, _brm(body))
    built = SBBRMI(Base.invokelatest(builder, data); mod=@__MODULE__, total_groups=())
    fresh = reprocess(sb, replay; freeze_constants=false)
    (; sb, built, fresh, rk=RKBRMI(brmi))
end

@stestset "constants of a body eval'd inside a function evaluate on both backends" begin
    conc = PANEL.conc[1:6] .* 10
    result = build_from_runtime_body(RUNTIME_BODY, PANEL, merge(NO_CONC, (; conc)))
    reference = SBBRMI(SUPPLIED(supplied(PANEL)); mod=@__MODULE__, total_groups=())
    for sb in (result.sb, result.built)
        @test constants(sb) == FITTED
        @test stan_code(sb) == stan_code(reference)
    end
    @test constants(result.fresh) == Dict(:ec50_loc => mean(log.(conc)),
        :ec50_scale => std(log.(conc)), :V => var(NO_CONC.y))
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = rk_logdensity_problem(result.rk; ad_backend)
    rk_supplied = rk_logdensity_problem(RKBRMI(SUPPLIED(supplied(PANEL))); ad_backend)
    for u in (zeros(LogDensityProblems.dimension(rk)), fill(0.3, LogDensityProblems.dimension(rk)))
        @test LogDensityProblems.logdensity(rk, u) == LogDensityProblems.logdensity(rk_supplied, u)
    end
end

refusal(f) = try
    f(); nothing
catch err
    sprint(showerror, err)
end

@stestset "constants refuse what is not one finite number of the data" begin
    # refused: a constant is computed before sampling, so it cannot read a
    # model value (decision `00kjor3`: data-calibrated scalars).
    message = refusal(() -> (@brm begin
        mu ~ 1 + x
        a = constant(mean(mu))
        y ~ Normal(mu, a)
    end)(PANEL))
    @test occursin("reads `mu`, a model value", message)
    message = refusal(() -> SBBRMI((@brm begin
        a = constant(log(-1.0 * first(conc)))
        y ~ Normal(0, a)
    end)(PANEL); mod=@__MODULE__))
    @test occursin("failed", message) && occursin("DomainError", message)
    # refused: a constant is data, so each value is one finite real number
    # (decision `00kjor3`); an array holds such numbers and nothing else.
    for expression in (:(fill(NaN, 3)), :(conc .> 1.0), :(reshape(conc .> 1.0, 3, 4)),
                       :(string(first(conc))), :(Inf))
        model = eval(:(@brm begin
            a = constant($expression)
            y ~ Normal(0, 1.0)
        end))
        message = refusal(() -> SBBRMI(model(PANEL); mod=@__MODULE__))
        @test occursin("one finite real number or an array of them", message)
    end
    message = refusal(() -> SBBRMI((@brm begin
        a = constant(mean(nonexistent_column))
        y ~ Normal(a, 1.0)
    end)(PANEL); mod=@__MODULE__))
    @test occursin("reads `nonexistent_column`", message)
    # refused: an unmarked whole-column reduction would broadcast over the
    # formula column; the declaration names it a constant (decision `00kjor3`).
    message = refusal(() -> (@brm begin
        a = mean(log.(conc))
        y ~ Normal(a, 1.0)
    end)(PANEL))
    @test occursin("constant(mean(log.(x)))", message)
    @test occursin("takes one expression", refusal(() -> eval(:(@brm begin
        a = constant(mean(conc), 2)
        y ~ Normal(a, 1.0)
    end))))
end
