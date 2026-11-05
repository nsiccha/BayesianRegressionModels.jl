include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_connected_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_producer_audit.jl"))
using LogDensityProblems, Random, Serialization
using ReactiveKernelsPPL: density_factors, select_factors, factor_coordinates

function check_separator_density(t)
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
        @test t.plan.point == t.m.unit
    end
end

function separator_simplex_model(data)
    @brm data begin
        composition ~ Dirichlet([1.0, 2.0, 1.5])
        mu ~ 1 + (1 | effect | subject)
        extra ~ Normal(mu[1] + composition[1], 0.5)
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu[i], 0.5)
        end
    end
end

@testset "explicit promotion closes a coupled prior and its transform" begin
    # Isolate prior/transform closure in one scientific subject. Hub and
    # chain fixtures below independently check the multi-subject regrouping.
    data = (; subject=fill("subject 1", 4), y=[-0.2, 0.1, 0.3, 0.5], extra=0.4)
    m = brm_nlme_model(RKBRMI(separator_simplex_model(data));
        ad_backend=NLME_AD, unit_partition=:connected)
    catalogue = density_factors(m.backend.model, BRM.rk_bound_plan(m.backend))
    block = factor_coordinates(select_factors(catalogue; kind=:prior, site=:composition))
    @test length(block) == 2
    original = BRM._brm_svi_target(m; separator=:none)
    @test all(j -> original.ownership.owner[j] == 1, block)
    promoted = BRM._brm_svi_target(m; separator=:none, global_coordinates=[first(block)])
    @test all(in(promoted.globals), block)
    @test length(promoted.units[1]) == length(original.units[1]) - 2
    check_separator_density(original)
    check_separator_density(promoted)
end

@testset "automatic hub separators rebuild exact inference partitions" begin
    n = 9
    data = connected_data(n, fill(1, n-1), collect(2:n))
    m = brm_nlme_model(RKBRMI(connected_model(data)); ad_backend=NLME_AD,
        unit_partition=:connected)
    scientific = deepcopy(m.view)
    hub = findall(==(1), m.view.coordinate_subject)
    none = BRM._brm_svi_target(m; separator=:none)
    manual = BRM._brm_svi_target(m; separator=:none,
        global_coordinates=m.view.coordinates[hub])
    # Splitting one unit also makes its shared population/scale parents global.
    # The budget counts the complete closure, not only the scientific cut.
    promotion_budget = length(setdiff(manual.globals, none.globals))
    @test promotion_budget > length(hub)
    limited = BRM._brm_svi_target(m; max_unit_coordinates=2,
        max_promoted_coordinates=1)
    @test limited.ownership.separator.status === :promotion_budget
    limited_fit = brm_svi_fit(m; max_unit_coordinates=2,
        max_promoted_coordinates=1, max_iterations=0, ndraws=0, threads=1)
    limited_carry = brm_svi_checkpoint(limited_fit)
    continued = brm_svi_fit(limited_carry, m; max_iterations=0, ndraws=0, threads=1)
    @test continued.diagnostics.globals == limited_fit.diagnostics.globals
    @test continued.diagnostics.units == limited_fit.diagnostics.units
    @test continued.diagnostics.q == limited_fit.diagnostics.q
    @test continued.diagnostics.separator.status === :carried
    @test continued.diagnostics.separator.trials == 0
    @test continued.diagnostics.separator.max_promoted_coordinates == 1
    t = BRM._brm_svi_target(m; max_unit_coordinates=2,
        max_promoted_coordinates=promotion_budget)
    println("SVI_HUB_PROMOTION before=", m.view.coordinates[none.globals],
        " after=", m.view.coordinates[t.globals], " budget=", promotion_budget)
    @test length(m.view.levels) == 1
    @test length(t.units) == n
    @test maximum(length, t.units) == 1
    @test all(j -> j in t.globals, hub)
    @test t.ownership.separator.status === :satisfied
    @test m.view.subject_coordinates == scientific.subject_coordinates
    @test m.view.subject_levels == scientific.subject_levels
    @test all(selection -> selection.globals === t.globals, t.plan.indices)
    @test all(i -> all(j -> t.ownership.owner[j] in (0, i), t.plan.footprints[i]),
        eachindex(t.units))
    @test manual.units == t.units && manual.globals == t.globals
    @test length(none.units) == 1
    audit = svi_producer_audit(t,
        [[offset + 0.12sin(j) for j in eachindex(m.unit)] for offset in (-0.31, 0.19)])
    @test length(audit.checks) == 2
    @test audit.units == t.units
    @test length(audit.factor_inventory) == length(t.ownership.factors.events)
    @test length(audit.likelihood_inventory) == sum(length, values(t.ownership.attribution))
    io = IOBuffer()
    Serialization.serialize(io, audit)
    seekstart(io)
    @test isequal(Serialization.deserialize(io), audit)
    @test t.plan.point == t.m.unit
    check_separator_density(manual)
    options = (; max_unit_coordinates=2, max_promoted_coordinates=promotion_budget,
        batch_size=4, max_iterations=9, ndraws=8, unit_dense_limit=0)
    full = brm_svi_fit(m; rng=Xoshiro(518), options..., threads=1)
    @test full.diagnostics.sampling.method === :work
    @test full.diagnostics.unit_levels == t.ownership.levels
    part = brm_svi_fit(m; rng=Xoshiro(518), options..., stop_after=3, threads=1)
    resumed = brm_svi_resume(m, part; ndraws=8, threads=1)
    @test svi_same_result(full, resumed)
    carry = brm_svi_checkpoint(full)
    @test carry.format == 3
    @test Set(keys(carry.units)) == Set(full.diagnostics.unit_levels)
    next = merge(data, (; y=data.y .+ 0.1))
    next_rk, _ = BRM._brm_svi_window_input(carry, connected_model(next), (;);
        caller="separator acceptance")
    next_model = brm_nlme_model(next_rk; ad_backend=NLME_AD,
        subject_group=carry.group, shared_groups=carry.shared,
        unit_partition=carry.unit_partition)
    next_target = BRM._brm_svi_target(next_model; previous=carry, carry.layout.options...)
    @test next_model.view.coordinates[next_target.globals] == carry.globals
    @test next_target.units == t.units
    fit = brm_svi_fit(carry, next_model; max_iterations=2, ndraws=0,
        batch_size=4, threads=1)
    @test fit.diagnostics.unit_levels == full.diagnostics.unit_levels
    @test fit.diagnostics.streaming.updates == 1
end

@testset "chain separators retain the native target" begin
    n = 17
    m = brm_nlme_model(RKBRMI(connected_model(connected_data(n,
        collect(1:n-1), collect(2:n)))); ad_backend=NLME_AD, unit_partition=:connected)
    # Three scientific cuts also promote their intercept and scale parents.
    cuts = findall(in((5, 10, 15)), m.view.coordinate_subject)
    manual = BRM._brm_svi_target(m; separator=:none, global_coordinates=cuts)
    original = BRM._brm_svi_target(m; separator=:none)
    budget = length(setdiff(manual.globals, original.globals))
    @test budget == 5
    @test maximum(length, manual.units) <= 4
    t = BRM._brm_svi_target(m; max_unit_coordinates=4, max_promoted_coordinates=budget)
    @test maximum(length, t.units) <= 4
    @test t.ownership.separator.status === :satisfied
    @test t.ownership.separator.trials <= 128
    check_separator_density(t)
end
