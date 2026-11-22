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

if ARGS == ["--write-fixture"]
    @assert pkgversion(BRM) == v"0.2.351"
    serialize(fixture, brm_fitted_state(RKBRMI(curves(training))))
    serialize(cell_fixture, brm_fitted_state(RKBRMI(offsets(cell_data))))
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
