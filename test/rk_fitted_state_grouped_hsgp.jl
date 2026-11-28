# The fixtures are complete format-2 exports from BRM 0.2.351 (8f1add0e).
# Regenerate them by running this file against that revision with --write-fixture.
using Test, Serialization, Distributions, BayesianRegressionModels
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels
const fixture = joinpath(@__DIR__, "fitted_state_programs", "grouped_hsgp_0.2.351.jls")
const cell_fixture = joinpath(@__DIR__, "fitted_state_programs", "cell_arithmetic_0.2.351.jls")

curves(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; by=site, k=2, c=1.5) + hsgp(w; by=batch, k=3, c=1.5) + (1 | batch)
    y ~ Normal(mu, sigma)
end
const training = (; site=repeat(["west", "east"], 6), batch=repeat([7, 2, 5], 4),
    x=collect(range(-1, 1; length=12)), w=sin.(1.0:12.0), y=cos.(1.0:12.0))

offsets(d) = @brm d begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(x)
        loc[i] = (a + x[i] + 0.5) - b
        y[i] ~ normal(loc[i], sigma)
    end
end
const cell_data = (; x=[[0.1, 0.2], [1.0]], y=[[0.3, 0.4], [0.8]])

# One response array per subject, observed in a plate the per-subject predictor
# labels by `subject`. `shifted_visits` adds an undotted `+`, which 0.2.351
# emitted as a bound helper.
visits(d) = @brm d begin
    sigma ~ Exponential(1)
    rate ~ 1 + weight + (1 | subject)
    effect(rate, Intercept) ~ Normal(0, 1)
    @plate for i in eachindex(time)
        loc[i] = exp.(-exp(rate[i]) .* time[i])
        y[i] ~ normal(loc[i], sigma)
    end
end
shifted_visits(d) = @brm d begin
    sigma ~ Exponential(1)
    rate ~ 1 + weight + (1 | subject)
    effect(rate, Intercept) ~ Normal(0, 1)
    @plate for i in eachindex(time)
        loc[i] = exp.(-exp(rate[i] + 0.5) .* time[i])
        y[i] ~ normal(loc[i], sigma)
    end
end
const plate_models = (; visits, shifted_visits)
const plate_data = (; subject=["s1", "s2", "s3"], weight=[61.0, 75.0, 68.0],
    time=[[0.5, 1.0], [0.5], [1.0, 2.0, 3.0]], y=[[0.9, 0.7], [1.4], [0.6, 0.3, 0.2]])
plate_fixture(name) = joinpath(@__DIR__, "fitted_state_programs", "labelled_plate_$(name)_0.2.351.jls")

# A subject plate that observes its rows of a long table through a `groups`
# binding, under a grouped HSGP over those rows whose index 0.2.351 wrote with
# its labels: the spelling before 0.2.363.
readings(d) = @brm d begin
    sigma ~ Exponential(1)
    base ~ 1 + (1 | subject)
    drift ~ 0 + hsgp(visit_time; by=visit_subject, k=2)
    visit_rows = groups(visit_subject => subject)
    @plate for i in eachindex(base)
        level[visit_rows[i]] ~ Normal(base[i] .+ drift[visit_rows[i]], sigma)
    end
end
const readings_data = (; subject=["a", "b", "c", "d"],
    visit_subject=["a", "b", "a", "c", "d", "b", "c", "a", "d", "c", "b", "d"],
    visit_time=[0.0, 0.5, 1.0, 0.2, 0.1, 1.5, 1.8, 2.5, 2.2, 3.0, 2.9, 3.4],
    level=[0.1, 0.9, 0.4, -0.2, 1.4, 0.6, 0.0, 0.8, 1.7, -0.4, 1.1, 1.5])

if ARGS == ["--write-fixture"]
    @assert pkgversion(BRM) == v"0.2.351"
    serialize(fixture, brm_fitted_state(RKBRMI(curves(training))))
    serialize(cell_fixture, brm_fitted_state(RKBRMI(offsets(cell_data))))
    for (name, build) in pairs(plate_models)
        serialize(plate_fixture(name), brm_fitted_state(RKBRMI(build(plate_data))))
    end
    serialize(plate_fixture(:readings), brm_fitted_state(RKBRMI(readings(readings_data))))
    exit()
end

function value_at(rk, draw, name)
    inputs = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec; bound=inputs,
        have=(:unconstrained, keys(inputs)...), want=(name,))
    Base.invokelatest(query, draw)
end

@testset "format-2 grouped HSGP from BRM 0.2.351" begin
    old = deserialize(fixture)
    fit = RKBRMI(curves(training))
    current = brm_fitted_state(fit)
    @test old.format == current.format == 2
    @test old.program != current.program
    @test !haskey(old.fitted.values, :site_fit_levels)
    @test old.fitted.values.batch_fit_levels == [2, 5, 7]
    @test old.fits[(:levels, :site)] == ["east", "west"]

    # First open the old file on a single fitted site, with the training
    # rows reversed. It must still address both sites' original draws.
    rows = [11, 7, 3]
    subset = map(col -> col[rows], training)
    draw = collect(range(-0.7, 0.8; length=length(old.coordinates)))
    replay = RKBRMI(curves(subset), old)
    oracle = RKBRMI(curves(subset), current)
    @test BRM._rk_layout_coordinate_names(replay) == old.coordinates
    bound, expected_bound = rk_bound_data(replay), rk_bound_data(oracle)
    @test keys(bound) == keys(expected_bound)
    # The old fit had no site_fit input or stored row count for it. Its
    # absent stand-in describes the saved level vector. All other entries,
    # including batch_fit (already present in the old fit), match exactly.
    @test size(bound.site_fit) == (2,)
    @test size(expected_bound.site_fit) == (12,)
    @test all(k -> isequal(bound[k], expected_bound[k]), setdiff(keys(bound), (:site_fit,)))
    @test value_at(replay, draw, :mu) == value_at(oracle, draw, :mu)
    exported = brm_fitted_state(replay)
    @test exported.program == current.program
    @test isequal(exported.fitted.values, current.fitted.values)
    @test exported.fit_twins == current.fit_twins
    @test exported.fitted_reads == current.fitted_reads

    # More sites than the fit and fewer batches. Both direct opening and a
    # later binder must keep track of which level sets are being resampled.
    design = merge(subset, (; site=["a", "b", "c"], batch=[9, 9, 9]))
    groups = (:site, :batch)
    fresh = RKBRMI(curves(design), old; resample_groups=groups)
    expected = RKBRMI(curves(design), current; resample_groups=groups)
    @test BRM._rk_layout_coordinate_names(fresh) == BRM._rk_layout_coordinate_names(expected)
    @test isequal(rk_bound_data(fresh), rk_bound_data(expected))
    for saved in (old, exported)
        binder = rk_design_binder(RKBRMI(curves(subset), saved); resample_groups=groups)
        reference = rk_design_binder(oracle; resample_groups=groups)
        @test binder.level_inputs == reference.level_inputs
        @test isequal(binder(design), reference(design))
        @test binder(design).site_fit_levels == ["east", "west"]
    end
    # Replay does not modify the caller's saved state.
    @test !haskey(old.fitted.values, :site_fit_levels)
    @test old.program == deserialize(fixture).program

    # A change to the old program (its weight extent, label order or law)
    # still fails the identity check, even when opening on resampled rows.
    for (before, after) in (("z[1:2,", "z[1:3,"),
            ("[\"east\", \"west\"]", "[\"west\", \"east\"]"),
            ("Normal.(0, 1)", "Normal.(1, 1)"))
        changed = deepcopy(old)
        i = findfirst(contains(before), changed.program)
        @test i !== nothing
        changed.program[i] = replace(changed.program[i], before => after; count=1)
        @test_throws "changes the fitted RK program" RKBRMI(curves(design), changed; resample_groups=groups)
    end
end

@testset "format-2 cell arithmetic from BRM 0.2.351" begin
    old = deserialize(cell_fixture)
    current = brm_fitted_state(RKBRMI(offsets(cell_data)))
    @test old.program != current.program
    @test "binding: _rk_kernel_stan_add" in old.program
    @test "binding: _rk_kernel_stan_sub" in old.program
    design = (; x=[[2.0], [-1.0, 0.0, 1.0]], y=[[0.1], [0.2, 0.3, 0.4]])
    replay = RKBRMI(offsets(design), old)
    oracle = RKBRMI(offsets(design), current)
    @test BRM._rk_layout_coordinate_names(replay) == old.coordinates
    @test isequal(rk_bound_data(replay), rk_bound_data(oracle))
    draw = collect(range(-0.6, 0.9; length=length(old.coordinates)))
    @test value_at(replay, draw, :loc) == value_at(oracle, draw, :loc)
    @test brm_fitted_state(replay).program == current.program
    # Both bindings still exist after swapping this operation. The whole
    # cell must match, including the operator used at each call site.
    changed = deepcopy(old)
    i = findfirst(contains("_rk_kernel_stan_sub("), changed.program)
    changed.program[i] = replace(changed.program[i], "_rk_kernel_stan_sub(" =>
        "_rk_kernel_stan_add("; count=1)
    @test_throws "changes the fitted RK program" RKBRMI(offsets(design), changed)
end

# 0.2.416 made a labelled plate's observation of one response array per cell
# loop over the column that labels the cells, where it looped over the response
# (snag `data-free-replay-5d3abb2c`: a downstream simulator's saved fits stopped
# loading). Both loops visit the same cells of a fit, so the stored statement is
# read in this BRM's form (`_rk_stored_program`), alone and together with the
# cell arithmetic helpers of the same export.
@testset "format-2 labelled plates from BRM 0.2.351" begin
    refusal(f) = try
        f(); nothing
    catch e
        e
    end
    over_response = "statement: #= line 0 =# @plate(for i = eachindex(y)\n"
    over_cells = "statement: #= line 0 =# @plate(for i = eachindex(subject)\n"
    for (name, build) in pairs(plate_models)
        old = deserialize(plate_fixture(name))
        current = brm_fitted_state(RKBRMI(build(plate_data)))
        @test old.format == current.format == 2
        # Only the loop of the observation differs, and the helpers where the
        # cell adds.
        k = findfirst(startswith(over_response), old.program)
        @test k !== nothing
        @test !any(startswith(over_response), current.program)
        @test replace(old.program[k], over_response => over_cells) in current.program
        @test ("binding: _rk_kernel_stan_add" in old.program) == (name === :shifted_visits)
        @test any(contains("stan_add(x, y)"), current.program) == (name === :shifted_visits)

        # Two of the fitted subjects in another order, with other visits.
        design = (; subject=["s3", "s1"], weight=[68.0, 61.0],
            time=[[0.25], [1.5, 2.5]], y=[[0.8], [0.5, 0.1]])
        replay = RKBRMI(build(design), old)
        oracle = RKBRMI(build(design), current)
        @test BRM._rk_layout_coordinate_names(replay) == old.coordinates
        @test isequal(rk_bound_data(replay), rk_bound_data(oracle))
        draw = collect(range(-0.6, 0.9; length=length(old.coordinates)))
        @test value_at(replay, draw, :loc) == value_at(oracle, draw, :loc)
        @test brm_fitted_state(replay).program == current.program
        # Replay does not modify the caller's saved state.
        @test old.program == deserialize(plate_fixture(name)).program

        # The observation is still compared: another law, or a loop over a
        # column that does not label the cells, is refused by that statement,
        # in the spelling the state was exported in.
        for (before, after) in (("Normal.(", "Cauchy.("), ("eachindex(y)", "eachindex(time)"))
            changed = deepcopy(old)
            @test occursin(before, changed.program[k])
            changed.program[k] = replace(changed.program[k], before => after; count=1)
            e = refusal(() -> RKBRMI(build(design), changed))
            @test e isa ErrorException
            @test occursin("changes the fitted RK program", e.msg)
            @test occursin("fitted: " * first(split(changed.program[k], '\n')), e.msg)
            @test !occursin("stan_add(x, y)", e.msg)
        end
    end

    # A state that also holds a grouped HSGP in the spelling before 0.2.363 is
    # compared with that spelling re-emitted, and its plate is read there too.
    # Refused for its plate, such a state named the `brm_grouped_hsgp`
    # definition, which this BRM writes with one more argument.
    @testset "with a grouped HSGP" begin
        old = deserialize(plate_fixture(:readings))
        current = brm_fitted_state(RKBRMI(readings(readings_data)))
        @test any(contains("brm_grouped_hsgp(axis, "), old.program)
        @test any(contains("brm_grouped_hsgp(fitted_levels, axis, "), current.program)
        # The observation is the one plate this BRM writes differently: the
        # same cell, looped over the subjects.
        plate = "statement: #= line 0 =# @plate(for i = eachindex("
        k = findfirst(s -> startswith(s, plate) && !(s in current.program), old.program)
        @test k !== nothing
        @test !startswith(old.program[k], over_cells)
        @test over_cells * last(split(old.program[k], '\n'; limit=2)) in current.program

        # Two of the fitted subjects in another order, with other visits.
        design = (; subject=["c", "a"], visit_subject=["a", "c", "a"],
            visit_time=[0.4, 1.2, 2.0], level=[0.3, -0.1, 0.7])
        replay = RKBRMI(readings(design), old)
        oracle = RKBRMI(readings(design), current)
        @test BRM._rk_layout_coordinate_names(replay) == old.coordinates
        draw = collect(range(-0.6, 0.9; length=length(old.coordinates)))
        @test value_at(replay, draw, :base) == value_at(oracle, draw, :base)
        @test value_at(replay, draw, :drift) == value_at(oracle, draw, :drift)
        @test brm_fitted_state(replay).program == current.program
        @test old.program == deserialize(plate_fixture(:readings)).program

        changed = deepcopy(old)
        changed.program[k] = replace(changed.program[k], "Normal.(" => "Cauchy.("; count=1)
        @test changed.program[k] != old.program[k]
        e = refusal(() -> RKBRMI(readings(design), changed))
        @test e isa ErrorException
        @test occursin("changes the fitted RK program", e.msg)
        @test occursin("fitted: " * first(split(changed.program[k], '\n')), e.msg)
        @test !occursin("brm_grouped_hsgp(", e.msg)
    end
end
