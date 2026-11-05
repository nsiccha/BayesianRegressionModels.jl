include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_producer_audit.jl"))
using ReactiveKernelsPPL: density_factors, select_factors, factor_coordinates,
    parameter_metadata

# Two independent scientific axes, as in the bounded ODL soil hierarchy.
# Soil cells have no trajectory parent; their shared-only likelihood is
# assigned once to inference unit 1. Its covariance may then be unit-owned.
function svi_atomic_partition_model(data)
    @brm data begin
        mu ~ 1 + (1 | trajectory | subject)
        soil_a ~ 1 + (1 | soil | sample)
        soil_b ~ 1 + (1 | soil | sample)
        cor(:, soil) ~ LKJCholesky(2, 2.0)
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu[i], 0.5)
        end
        @plate for i in eachindex(soil_y)
            soil_y[i] ~ Normal(soil_a[i] + soil_b[i], 0.5)
        end
    end
end

@stestset "a local atomic correlation preserves its packed partition ownership" begin
    data = (; subject=["trajectory 1", "trajectory 2"],
        sample=["soil 1", "soil 2"], y=[0.2, -0.1], soil_y=[0.3, 0.4])
    rk = RKBRMI(svi_atomic_partition_model(data))
    m = brm_nlme_model(rk; ad_backend=NLME_AD,
        unit_partition=:connected, shared_groups=(:sample,))
    catalogue = density_factors(rk.model, BRM.rk_bound_plan(rk))
    event = only(filter(collect(catalogue)) do event
        event.kind === :prior && endswith(string(event.site), ".L")
    end)
    block = factor_coordinates(select_factors(catalogue; kind=:prior, site=event.site))
    metadata = only(filter(parameter_metadata(rk.model)) do entry
        endswith(join(string.(entry.path), "."), ".L")
    end)
    @test metadata.shape == (2, 2)
    @test metadata.ncoordinates == length(block) == 1
    @test all(iszero, m.view.coordinate_subject[block])
    original = deepcopy(m.view)
    ownership = BRM._brm_svi_connected_ownership(m; separator=:auto)
    @test all(==(1), ownership.owner[block])
    declarations = Dict(BRM._rk_nlme_partitioned(rk, ownership.owner))
    @test declarations[event.site] == [1, 1]
    target = BRM._brm_svi_target(m; separator=:auto)
    @test all(==(1), target.ownership.owner[block])
    @test m.view.subject_levels == original.subject_levels
    @test m.view.coordinate_subject == original.coordinate_subject
    promoted = BRM._brm_svi_target(m; separator=:none,
        global_coordinates=m.view.coordinates[block])
    @test all(in(promoted.globals), block)
    points = [[offset + 0.12sin(j) for j in eachindex(m.unit)]
        for offset in (-0.31, 0.19)]
    for t in (target, promoted)
        audit = svi_producer_audit(t, points)
        @test length(audit.checks) == 2
        @test all(check -> check.gradient_max_error < 2e-8, audit.checks)
        @test count(item -> item.kind === :prior && item.site == event.site,
            audit.factor_inventory) == 1
    end
end
