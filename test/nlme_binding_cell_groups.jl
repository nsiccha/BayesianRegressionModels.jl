# test/nlme_binding_cell_groups.jl — `brm_nlme_model`'s subject cell groups
# are the program's, not the data's (snag odl-s-ladder-brm-d114c22c). Two
# bindings of one program: paths (the subjects, one grouping row each), point
# measurements on rows of their own that read the level of the path they were
# taken on, and readings of the first paths, each reading its path's level
# through an index (the shape of a downstream model whose later paths carry
# point measurements only). In the first binding
# every path has a reading and there are no point measurements, so the
# readings have as many densities as the grouping has rows and are attributed
# by those rows; in the second, one path has no reading, so the readings are
# measured. The groups, their order, the partition's keys and the model's type
# used to follow that (the readings joined the plate group, placed first);
# now they are the same for both bindings, so the second reuses the code the
# first compiled.
#
# Covered: one group per authored plate, in the program's pointwise order, for
# both bindings; the same partition keys and model type; how each binding
# attributed the readings; lockstep values, per-subject cells and unit problems
# against the full RK density on both.
#
# Run: julia --project=test test/nlme_binding_cell_groups.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random
include(joinpath(@__DIR__, "nlme_unit_checks.jl"))

function binding_cells_data(; npath, nread, spot_path)
    rng = Xoshiro(npath)
    (; path=["p$g" for g in 1:npath], reading=randn(rng, nread),
       reading_path=collect(1:nread), spot_path, spot_value=randn(rng, length(spot_path)))
end

binding_cells_model(data) = @brm data begin
    sigma ~ Exponential(1)
    level ~ 1 + (1 | path)
    @plate for j in eachindex(spot_value)
        spot_value[j] ~ Normal(level[spot_path[j]], sigma)
    end
    @plate for g in eachindex(reading)
        reading[g] ~ Normal(level[reading_path[g]], sigma)
    end
end

@stestset "one program's bindings have the same cell groups and model type" begin
    rows = binding_cells_data(; npath=2, nread=2, spot_path=Int[])
    measured = binding_cells_data(; npath=4, nread=3, spot_path=[2, 4, 4])
    models = map((rows, measured)) do data
        brm_nlme_model(RKBRMI(binding_cells_model(data)); ad_backend=NLME_AD,
            subject_group=:path)
    end
    a, b = models
    # The first binding reads the readings' subjects from the rows (a plate
    # over the subjects' rows), the second measures them.
    readings(m) = only(c for c in m.cell_query if c.names == (:reading,))
    @test !readings(a).elementwise && readings(b).elementwise
    @test [only(c) for c in readings(a).positions] == 1:2
    @test [length(p) for p in readings(b).positions] == [1, 1, 1, 0]
    @test [length(p) for p in only(c for c in b.cell_query
        if c.names == (:spot_value,)).positions] == [0, 1, 0, 2]
    # One group per plate, in the program's pointwise order, for both.
    order = collect(keys(Base.invokelatest(pointwise_query(a.backend), a.unit)))
    for m in models
        @test [only(c.names) for c in m.cell_query] == order
        @test collect(keys(m.cell_gradient.context.by)) == order
    end
    @test a.cell_gradient.context.by == b.cell_gradient.context.by
    @test typeof(a) == typeof(b)
    for (k, m) in enumerate(models)
        θ, σ, H, values = check_lockstep(m, Xoshiro(20 + k))
        for i in eachindex(m.view.levels)
            @test brm_nlme_loglikelihood(m, i, θ, σ, column(H, i)) ≈ values[i] rtol=1e-10
        end
        check_unit_problems(m, Xoshiro(30 + k))
    end
end

println("nlme_binding_cell_groups.jl: all testsets passed")
