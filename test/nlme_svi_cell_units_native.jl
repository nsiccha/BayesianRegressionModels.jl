# Connected SVI's `cell_units` on real models: what the units hold, that their
# terms still sum to the full density and gradient, and that fits, checkpoints and
# streaming windows work over them.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "..", "research", "svi_cell_units", "toy.jl"))
using LogDensityProblems, Random

function check_cell_unit_density(t)
    full = rk_logdensity_problem(t.m.backend; ad_backend=NLME_AD)
    for offset in (-0.31, 0.19)
        u = [offset + 0.12sin(j) for j in eachindex(t.m.unit)]
        g = u[t.globals]
        value, global_gradient = BRM._brm_svi_global_term(t, g)
        gradient = zeros(length(u))
        gradient[t.globals] .= global_gradient
        for (i, block) in enumerate(t.units)
            gg, gz = zeros(length(g)), zeros(length(block))
            value += BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[block])
            gradient[t.globals] .+= gg
            gradient[block] .+= gz
        end
        expected, native = LogDensityProblems.logdensity_and_gradient(full, u)
        @test value ≈ expected atol=2e-9 rtol=2e-10
        @test gradient ≈ native atol=2e-8 rtol=2e-9
    end
end

toy_model(; nsite=4, nsensor=6, sensor_rows=8, site_rows=[3, 6], seed=11) = let
    data = cell_toy_data(Xoshiro(seed); nsite, nsensor, nfield=6, nread=2, site_rows, sensor_rows)
    m = brm_nlme_model(RKBRMI(cell_toy_model(data)); ad_backend=NLME_AD,
        subject_group=:site, unit_partition=:connected)
    (; data, m, globals=cell_toy_globals(m))
end

names_of(m, block) = String.(m.view.coordinates[block])

@stestset "each sensor's effects and rows form a unit; the first unit no longer holds them" begin
    (; data, m, globals) = toy_model()
    first_unit = BRM._brm_svi_target(m; separator=:none, global_coordinates=globals,
        cell_units=:first_unit)
    split = BRM._brm_svi_target(m; separator=:none, global_coordinates=globals)
    @test split.ownership.separator.cell_units === :components
    # The reported layout, unchanged under `:first_unit`.
    @test length(first_unit.units) == 4
    @test count(n -> occursin("_z", n), names_of(m, first_unit.units[1])) == 12
    # Scientific units first, in order, then one unit per sensor.
    @test length(split.units) == 4 + 6
    @test split.ownership.levels[1:4] == first_unit.ownership.levels[1:4]
    @test split.ownership.levels[1:4] == [(s,) for s in data.site]
    @test length.(split.units[1:4]) == fill(length(split.units[2]), 4)
    @test all(==(2), length.(split.units[5:end]))
    @test sort(reduce(vcat, split.units[5:end])) == sort(setdiff(first_unit.units[1], split.units[1]))
    @test !any(n -> occursin("_z", n), names_of(m, split.units[1]))
    for i in 5:10
        z = names_of(m, split.units[i])
        @test count(n -> startswith(n, "gain_z"), z) == 1
        @test count(n -> startswith(n, "scale_z"), z) == 1
    end
    # A row reads exactly its own sensor's coordinates among the local ones.
    rows = Dict(split.ownership.attribution)[:row_y]
    @test length(rows) == length(data.row_y)
    for s in 1:6
        @test length(unique(rows[data.row_sensor .== s])) == 1
    end
    # Unit 1's work falls to what any other site with its rows has.
    @test first_unit.ownership.work[1] > 5maximum(first_unit.ownership.work[2:end])
    @test split.ownership.work[1] == split.ownership.work[3]
    check_cell_unit_density(first_unit)
    check_cell_unit_density(split)
end

@stestset "without promoted hyperparameters the sensors join, and a separator splits them" begin
    (; m) = toy_model()
    joined = BRM._brm_svi_target(m; separator=:none)
    @test length(joined.units) == 5
    @test count(n -> occursin("_z", n), names_of(m, joined.units[5])) == 12
    check_cell_unit_density(joined)
    # Automatic promotion of the hubs splits them when a budget asks for it.
    automatic = BRM._brm_svi_target(m; separator=:auto, max_unit_coordinates=6)
    @test automatic.ownership.separator.status === :satisfied
    @test length(automatic.units) > 5
    check_cell_unit_density(automatic)
end

# Rows of a family that reads only globals have no factor of their own to share.
function global_rows_data(n, m)
    (; subject=["s$i" for i in 1:n], y=[[0.1i, -0.2] for i in 1:n],
        extra=collect(range(-1.0, 1.0; length=m)))
end
global_rows_model(data) = @brm data begin
    sigma ~ Exponential(1)
    center ~ Normal(0, 1)
    mu ~ 1 + (1 | effect | subject)
    @plate for i in eachindex(mu)
        y[i] ~ Normal(mu[i] .+ center, sigma)
    end
    extra ~ Normal(center, sigma)
end

@stestset "rows reading only globals are grouped into zero-dimensional units" begin
    data = global_rows_data(3, 16)
    m = brm_nlme_model(RKBRMI(global_rows_model(data)); ad_backend=NLME_AD,
        subject_group=:subject, unit_partition=:connected)
    t = BRM._brm_svi_target(m; separator=:none)
    @test length(t.units) == 3 + 4
    @test all(isempty, t.units[4:end])
    rows = Dict(t.ownership.attribution)[:extra]
    @test all(i -> count(==(i), rows) == 4, 4:7)
    @test only.(t.ownership.levels[4:end]) == [(:cells, k) for k in 1:4]
    check_cell_unit_density(t)
    fit = brm_svi_fit(m; rng=Xoshiro(1), batch_size=3, max_iterations=40, window=10,
        separator=:none, ndraws=8, threads=1)
    @test all(isfinite, fit.draws)
    @test all(isfinite, fit.diagnostics.elbo)
    @test length(fit.diagnostics.unit_levels) == 7
    @test sum(fit.diagnostics.sampling.draws[4:end]) > 0
    @test fit.diagnostics.separator.cell_units === :components
end

@stestset "fits, resume and the old layout's checkpoints" begin
    (; m, globals) = toy_model()
    options = (; batch_size=3, window=10, separator=:none, global_coordinates=globals,
        threads=1, ndraws=8)
    straight = brm_svi_fit(m; rng=Xoshiro(2), max_iterations=30, options...)
    @test all(isfinite, straight.draws)
    @test length(straight.diagnostics.unit_levels) == 10
    @test all(>(0), straight.diagnostics.sampling.probabilities)
    # A resumed run equals the straight one, with the layout read from the checkpoint.
    partial = brm_svi_fit(m; rng=Xoshiro(2), max_iterations=30, stop_after=12, options...)
    state = brm_svi_optimizer_checkpoint(partial)
    resumed = brm_svi_resume(m, state; threads=1, ndraws=8)
    @test resumed.diagnostics.elbo == straight.diagnostics.elbo
    @test resumed.draws == straight.draws
    # A checkpoint of the first-unit layout stays on it, without being told.
    old = brm_svi_fit(m; rng=Xoshiro(2), max_iterations=30, stop_after=12,
        cell_units=:first_unit, options...)
    @test !hasproperty(old.diagnostics.optimizer.identity.separator, :cell_units)
    @test length(old.diagnostics.unit_levels) == 4
    continued = brm_svi_resume(m, brm_svi_optimizer_checkpoint(old); threads=1, ndraws=8)
    @test length(continued.diagnostics.unit_levels) == 4
    @test continued.diagnostics.iterations == 30
    # Sampling stays unbiased: draws over batch and probabilities are the fit's own.
    @test_throws ArgumentError brm_svi_fit(m; max_iterations=1, cell_units=:each_row)
end

@stestset "a streaming window keeps cell units by their coordinates" begin
    # Two windows of one model: sensor 3 reports only in the first, sensor 5 only
    # in the second, the others in both, with disjoint rows.
    data = cell_toy_data(Xoshiro(21); nsite=3, nsensor=5, nfield=6, nread=2,
        site_rows=[4], sensor_rows=6)
    window(keep) = let rows = [k for k in eachindex(data.row_sensor) if keep(data.row_sensor[k], k)]
        merge(data, (; row_sensor=data.row_sensor[rows], row_state=data.row_state[rows],
            row_weight=data.row_weight[rows], row_y=data.row_y[rows],
            row_index=collect(eachindex(rows))))
    end
    first_window = window((s, k) -> s != 5 && isodd(k))
    second_window = merge(window((s, k) -> s != 3 && iseven(k)), (;
        site=["later $g" for g in 1:3]))
    m1 = brm_nlme_model(RKBRMI(cell_toy_model(first_window)); ad_backend=NLME_AD,
        subject_group=:site, unit_partition=:connected)
    globals = cell_toy_globals(m1)
    options = (; batch_size=2, window=5, separator=:none, global_coordinates=globals,
        threads=1, ndraws=6)
    first_fit = brm_svi_fit(m1; rng=Xoshiro(3), max_iterations=10, options...)
    labels = first_fit.diagnostics.unit_levels
    @test count(level -> first(only(level)) === :cells, labels[4:end]) == length(labels) - 3
    state = brm_svi_checkpoint(first_fit)
    @test state.layout.options.cell_units === :components
    # Coordinates of sensor 5 have no reader in the first window: they stay global.
    # A later reader of a carried global cannot make it local.
    @test any(n -> startswith(String(n), "gain_z.5") || startswith(String(n), "gain_z[5"),
        state.globals)
    m2 = brm_nlme_model(RKBRMI(cell_toy_model(second_window)); ad_backend=NLME_AD,
        subject_group=:site, unit_partition=:connected)
    second_fit = brm_svi_fit(state, m2; rng=Xoshiro(4), max_iterations=10,
        batch_size=2, window=5, threads=1, ndraws=6)
    @test all(isfinite, second_fit.draws)
    carried = second_fit.diagnostics.unit_levels
    # The sensors that report again are the same units, restored rather than restarted.
    @test !isempty(intersect(labels[4:end], carried))
    # The sensor absent from this window keeps its factor in the next checkpoint.
    next = brm_svi_checkpoint(second_fit)
    @test all(label -> haskey(next.units, label), labels)
end

@stestset "`work` counts the rows of a vector-valued cell" begin
    (; data, m, globals) = toy_model(; nsite=4, site_rows=[3, 30])
    t = BRM._brm_svi_target(m; separator=:none, global_coordinates=globals)
    short, long = findall(==(3), length.(data.site_y)), findall(==(30), length.(data.site_y))
    @test !isempty(short) && !isempty(long)
    # Reads alone are the same for short and long sites; rows are what separates them.
    @test length(t.plan.footprints[short[1]]) == length(t.plan.footprints[long[1]])
    @test t.ownership.work[long[1]] - t.ownership.work[short[1]] ≈
        27 * BRM._BRM_SVI_ROW_WORK
    # An elementwise row adds no row term.
    reads = Dict(t.ownership.attribution)[:row_y]
    sensor = reads[1]
    inventory = BRM._brm_svi_factor_inventory(m.view, length(m.unit))
    rows_of = [i for (i, source) in enumerate(inventory.sources)
        if source[1] === :row_y && reads[source[2]] == sensor]
    @test t.ownership.work[sensor] >= sum(1 + length(inventory.supports[i]) for i in rows_of)
end

@stestset "`:components` is connected mode's alone" begin
    data = global_rows_data(3, 4)
    m = brm_nlme_model(RKBRMI(global_rows_model(data)); ad_backend=NLME_AD,
        subject_group=:subject)
    @test_throws ArgumentError BRM._brm_svi_target(m; cell_units=:components)
    t = BRM._brm_svi_target(m; cell_units=:first_unit)
    @test length(t.units) == 3
end
