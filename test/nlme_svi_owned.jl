using Test, Random, LinearAlgebra, Serialization
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface
using ADTypes: AutoEnzyme
import Enzyme
include("nlme_svi_owned_fixture.jl")
const BRM = BayesianRegressionModels
const OWNED_AD = AutoEnzyme(; mode=Enzyme.Reverse)

function owned_model(data; connected=false)
    rk = RKBRMI(connected ? svi_owned_connected_model(data) : svi_owned_model(data))
    brm_nlme_model(rk; ad_backend=OWNED_AD,
        unit_partition=connected ? :connected : :subjects)
end

function check_owned_density(t)
    m = t.m
    sampler = prepare_sampler(m.backend.model, rk_bound_plan(m.backend), m.unit;
        backend=OWNED_AD)
    for offset in (-0.23, 0.0, 0.31)
        u = [0.12sin(i) + offset for i in eachindex(m.unit)]
        g = u[t.globals]
        value, gg = BRM._brm_svi_global_term(t, g)
        global_value, global_gradient = value, copy(gg)
        unit_values, unit_gradients = Float64[], Vector{Float64}[]
        grad = zeros(length(u))
        grad[t.globals] .= gg
        for (i, block) in enumerate(t.units)
            gg, gz = zeros(length(g)), zeros(length(block))
            term = BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[block])
            value += term
            grad[t.globals] .+= gg
            grad[block] .+= gz
            packed = zeros(length(u))
            packed[t.globals] .= gg
            packed[block] .= gz
            push!(unit_values, term)
            push!(unit_gradients, packed)
        end
        expected_grad = similar(u)
        expected, _ = sampler_value_and_gradient!(sampler, expected_grad, u)
        @test value ≈ expected atol=2e-10 rtol=2e-12
        @test grad ≈ expected_grad atol=2e-9 rtol=2e-10
        # Average all singleton minibatches: each complete unit term,
        # including its authored priors, receives exactly the N/B weight.
        average = global_value + sum(length(t.units) .* unit_values) / length(t.units)
        average_gradient = zeros(length(u))
        average_gradient[t.globals] .= global_gradient
        average_gradient .+= sum(length(t.units) .* unit_gradients) ./ length(t.units)
        @test average ≈ expected atol=2e-10 rtol=2e-12
        @test average_gradient ≈ expected_grad atol=2e-9 rtol=2e-10
    end
end

@testset "ordinary coordinate ownership uses dormant structural reads" begin
    data = svi_owned_data()
    m = owned_model(data)
    t = BRM._brm_svi_target(m)
    @test length(t.globals) == 4
    @test length.(t.units) == fill(3, 6)
    check_owned_density(t)
    shared = BRM._brm_svi_target(owned_model(svi_owned_data(; shared=true)))
    # z[1] is shared by all six cells, even though its branch is dormant.
    # The unused odd coordinates are prior-only globals too.
    @test length(shared.globals) == 10
    @test length.(shared.units) == fill(2, 6)
    check_owned_density(shared)
    forced = last(t.units[1])
    promoted = BRM._brm_svi_target(m; global_coordinates=[m.view.coordinates[forced]])
    @test forced in promoted.globals
    @test length(promoted.units[1]) == 2
    check_owned_density(promoted)
    effect = first(m.view.subject_coordinates[1])
    promoted_effect = BRM._brm_svi_target(m; global_coordinates=[effect])
    @test effect in promoted_effect.globals
    @test length(promoted_effect.units) == length(t.units)
    check_owned_density(promoted_effect)
    # A fully promoted unit still contributes its likelihood to the batch,
    # with an empty local Gaussian and no local natural-gradient step.
    all_promoted = brm_svi_fit(m; global_coordinates=t.units[1], rng=Xoshiro(40),
        batch_size=6, max_iterations=1, ndraws=2, threads=1)
    @test isempty(all_promoted.diagnostics.units[1])
    @test all(isfinite, all_promoted.draws)
    options = (; batch_size=2, max_iterations=4, ndraws=4, threads=1,
        global_coordinates=[m.view.coordinates[forced]])
    full = brm_svi_fit(m; rng=Xoshiro(41), options...)
    part = brm_svi_fit(m; rng=Xoshiro(41), stop_after=2, options...)
    resumed = brm_svi_resume(m, brm_svi_optimizer_checkpoint(part); ndraws=4, threads=1)
    @test resumed.draws == full.draws
    @test resumed.diagnostics.q == full.diagnostics.q
    if Threads.nthreads() > 1
        threaded = brm_svi_fit(m; rng=Xoshiro(41),
            merge(options, (; threads=2))...)
        @test threaded.draws == full.draws
        @test threaded.diagnostics.q == full.diagnostics.q
    end
    mktempdir() do directory
        input, output = joinpath(directory, "input.jls"), joinpath(directory, "output.jls")
        serialize(input, (; data, state=brm_svi_optimizer_checkpoint(part)))
        script = joinpath(@__DIR__, "nlme_svi_owned_process.jl")
        run(`$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $script $input $output`)
        fresh = deserialize(output)
        @test fresh.draws == full.draws
        @test fresh.q == full.diagnostics.q
    end
end

@testset "connected units own ordinary coordinates without changing scientific roles" begin
    data = merge(svi_owned_data(), (; left=collect(1:6), right=[2, 1, 3, 4, 5, 6],
        weight=zeros(6)))
    m = owned_model(data; connected=true)
    t = BRM._brm_svi_target(m)
    @test m.view.unit_members == [[1, 2], [3], [4], [5], [6]]
    @test length(t.globals) == 4
    @test length.(t.units) == [6, 3, 3, 3, 3]
    @test count(==(:population), m.view.role) == 13
    check_owned_density(t)
    fit = brm_svi_fit(m; rng=Xoshiro(42), batch_size=2, max_iterations=2,
        ndraws=3, threads=1)
    @test fit.diagnostics.units == t.units
    @test size(fit.draws) == (length(m.unit), 3)
end

@testset "owned coordinates survive absent and returning streaming units" begin
    data = svi_owned_data()
    initial = brm_svi_fit(svi_owned_model(data); rng=Xoshiro(51), max_iterations=2,
        ndraws=0, threads=1)
    state = brm_svi_checkpoint(initial)
    @test all(u -> length(u.keys) == 3, values(state.units))
    next_data = merge(svi_owned_data([6, 2, 4]), (; ncoeff=12))
    update = brm_svi_fit(state, svi_owned_model(next_data); max_iterations=0,
        ndraws=0, threads=1)
    carried = brm_svi_checkpoint(update)
    @test carried.units == state.units
    @test carried.q == state.q
    @test length(update.diagnostics.globals) == 4
    output_state = deepcopy(state)
    for label in ("s1", "s3", "s5")
        output_state.units[label].M .= [0.3, 1.7, -0.8]
        output_state.units[label].U .*= 1e100
    end
    output_target = BRM._brm_svi_target(update.diagnostics.model; previous=output_state)
    output_draws = BRM._brm_svi_draws(Xoshiro(53), output_target, update.diagnostics.q, 2)
    names = update.coordinates
    for label in ("s1", "s3", "s5"), (k, key) in enumerate(output_state.units[label].keys)
        hasproperty(key, :coordinate) || continue
        j = findfirst(==(key.coordinate), names)
        @test all(value -> value ≈ output_state.units[label].M[k], output_draws[j, :])
    end
    returned = brm_svi_fit(carried, svi_owned_model(data); max_iterations=0,
        ndraws=0, threads=1)
    @test brm_svi_checkpoint(returned).units == state.units
    new_data = svi_owned_data([6, 2, 4, 7])
    new_fit = brm_svi_fit(carried, svi_owned_model(new_data); max_iterations=0,
        ndraws=0, threads=1)
    new_state = brm_svi_checkpoint(new_fit)
    @test haskey(new_state.units, "s7")
    @test length(new_state.units["s7"].keys) == 3
    @test length(new_fit.diagnostics.globals) == 4
    new_model = new_fit.diagnostics.model
    new_target = BRM._brm_svi_target(new_model; previous=carried)
    stream_target, _ = BRM._brm_svi_initialize(new_target, carried, carried.q.family;
        start=nothing, inits=nothing, global_scale=0.1, unit_scale=1.0)
    new_i = findfirst(==("s7"), new_model.view.levels)
    g, z = fill(0.2, length(new_target.globals)), fill(0.7, length(new_target.units[new_i]))
    gg, gz = zeros(length(g)), zeros(length(z))
    expected_gg, expected_gz = copy(gg), copy(gz)
    @test BRM._brm_svi_unit_prior!(gg, gz, stream_target.prior, stream_target, new_i, g, z) ≈
        BRM._brm_svi_unit_prior!(expected_gg, expected_gz, new_target.prior, new_target, new_i, g, z)
    @test gg ≈ expected_gg
    @test gz ≈ expected_gz
    m = update.diagnostics.model
    t0 = BRM._brm_svi_target(m; previous=state)
    t, _ = BRM._brm_svi_initialize(t0, state, state.q.family;
        start=nothing, inits=nothing, global_scale=0.1, unit_scale=1.0)
    # The carried Gaussian replaces each complete authored bucket. It must
    # not add the original Normal(0.4,tau) priors a second time.
    g = fill(0.2, length(t.globals))
    for (i, label) in enumerate(m.view.levels)
        z = fill(0.7, length(t.units[i]))
        gg, gz = zeros(length(g)), zeros(length(z))
        value = BRM._brm_svi_unit_prior!(gg, gz, t.prior, t, i, g, z)
        old = state.units[label]
        delta = z - old.M
        @test value ≈ -dot(delta, old.Λ * delta) / 2 +
            logdet(Symmetric(old.Λ)) / 2 - length(z) * log(2pi) / 2
        @test gz ≈ -old.Λ * delta
        @test iszero(gg)
    end
end
