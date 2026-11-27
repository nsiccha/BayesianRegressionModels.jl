module CollectionSubjectGatherTest
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernelsDistributionKernels

const ordinal = @brm begin
    gain ~ Normal(0, 1)
    @plate for i in eachindex(y)
        mu[i] = gain * i
        y[i] ~ Normal(mu[i], 1)
    end
end
@stestset "An ordinal plate index is a cell value" begin
    data = (; y=[.2, -.1, .3])
    brmi = ordinal(data)
    rk, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "plate-ordinal-index"; mod=@__MODULE__)
    @test BridgeStan.param_unc_num(stan.model) == 1
    oracle(u) = logpdf(Normal(), u[1]) +
        sum(logpdf.(Normal.(u[1] .* [1, 2, 3], 1), data.y))
    for point in ([0.], [.3])
        value, gradient = check_consumer_point(problem, point, oracle)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model,
            point; propto=false)
        @test value ≈ stan_value
        @test gradient ≈ stan_gradient
    end
end

@brm function reading(entry)
    bias ~ Normal(0, 1)
    if entry.series
        value = bias .+ entry.grid
    else
        value = bias + entry.level
    end
    return (; value)
end
const gathered = @brm begin
    for entry in records
        channels[entry.id] ~ reading(entry)
    end
    values = channels.value
    @plate for i in eachindex(y)
        selected = [length(v) == 1 ? v[1] : v[i] for v in values]
        shared[i] = sum(weights .* selected)
        y[i] ~ Normal(shared[i], 1)
    end
end

function check_gather(label, data)
    brmi = gathered(data)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(brmi, "collection-subject-" * label; mod=@__MODULE__)
    names = BridgeStan.param_unc_names(stan.model)
    @test length(names) == 2
    descriptor = brm_descriptor(sb)
    indices = Dict(record.id => only(brm_output_coordinates(descriptor, :bias, names;
        submodel=(:channels, record.id))) for record in data.records)
    @test sort!(collect(values(indices))) == [1, 2]
    function means(u, design=data)
        result = zeros(length(design.y))
        for (record, weight) in zip(design.records, design.weights)
            result .+= weight .* (u[indices[record.id]] .+
                (record.series ? record.grid : record.level))
        end
        result
    end
    oracle(u, design=data) = sum(logpdf.(Normal(), u)) +
        sum(logpdf.(Normal.(means(u, design), 1), design.y))
    rk, problem = consumer_problem(brmi)
    transport = brm_coordinate_transport(rk, sb, names)
    for u in (zeros(2), [.2, -.3])
        expected(v) = oracle(brm_rk_point_to_stan(transport, v))
        value, gradient = check_consumer_point(problem, u, expected)
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_u; propto=false)
        @test stan_value ≈ oracle(stan_u)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value)
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient)
        draws = rk_draw_values(rk, reshape(u, :, 1), (:values, :shared))
        @test only(draws.shared) ≈ means(stan_u)
        for (j, record) in enumerate(data.records)
            expected_value = stan_u[indices[record.id]] .+
                (record.series ? record.grid : record.level)
            @test only(draws.values)[j] ≈ expected_value
            member = rk_draw_values(rk, reshape(u, :, 1), (:value,);
                submodel=(:channels, record.id))
            @test only(member.value) ≈ expected_value
        end
    end
    # Reorder keys and change the subject count without changing fitted coordinates.
    changed = (; data..., y=[.1, -.4, .2, .3], weights=[.7, -.2],
        records=[(; record..., grid=record.series ? [.4, .7, -.1, .8] : Float64[])
            for record in reverse(data.records)])
    replay = reprocess(rk, changed)
    @test coordinate_names(replay.model.layout) == coordinate_names(rk.model.layout)
    point = [.2, -.3]
    replay_problem = rk_logdensity_problem(replay; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    check_consumer_point(replay_problem, point,
        u -> oracle(brm_rk_point_to_stan(transport, u), changed))
    @test only(rk_draw_values(replay, reshape(point, :, 1), (:shared,)).shared) ≈
        means(brm_rk_point_to_stan(transport, point), changed)
end

@stestset "A parent gathers keyed values at its subject index" begin
    vector_b = (; id=:b, series=true, grid=[.5, 1.2, -.3], level=0.)
    vector_a = (; id=:a, series=true, grid=[1.1, -.2, .7], level=0.)
    scalar_b = (; id=:b, series=false, grid=Float64[], level=.4)
    scalar_a = (; id=:a, series=false, grid=Float64[], level=-.3)
    for (label, records) in (("mixed", [vector_b, scalar_a]),
            ("vectors", [vector_b, vector_a]), ("scalars", [scalar_b, scalar_a]))
        @testset "$label" begin
            check_gather(label, (; records, weights=[.3, .8], y=[.2, -.1, .3]))
        end
    end
end

const scoped = @brm begin
    gain ~ Normal(0, 1)
    @plate for i in eachindex(y)
        selected = [v[i] + x[i] for v in catalog]
        shift = sum([i for i in offsets])
        loc[i] = gain * sum(selected) + shift
        y[i] ~ Normal(loc[i], 1)
    end
end

@stestset "Inner bindings shadow data and the enclosing index" begin
    data = (; catalog=[[.2, .4, .7], [.5, -.3, .6]], x=[.1, -.1, .2],
        offsets=[.2, .3], y=[.3, -.1, .4], v=[100., 200., 300., 400.])
    base = [sum(curve[j] for curve in data.catalog) + 2 * data.x[j] for j in 1:3]
    oracle(u) = logpdf(Normal(), u[1]) +
        sum(logpdf.(Normal.(u[1] .* base .+ sum(data.offsets), 1), data.y))
    brmi = scoped(data)
    rk, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "subject-gather-lexical-scope"; mod=@__MODULE__)
    @test BridgeStan.param_unc_num(stan.model) == 1
    for point in ([0.], [.3])
        value, gradient = check_consumer_point(problem, point, oracle)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, point; propto=false)
        @test value ≈ stan_value
        @test gradient ≈ stan_gradient
    end
end

@brm function endpoint(entry, shared)
    bias ~ Normal(0, 1)
    @plate for i in eachindex(entry.y)
        loc[i] = entry.weight * shared[i] + bias
        entry.y[i] ~ Normal(loc[i], 1)
    end
    return (; loc)
end
const pipeline = @brm begin
    for entry in records
        channels[entry.id] ~ reading(entry)
    end
    values = channels.value
    @plate for i in eachindex(subjects)
        selected = [length(v) == 1 ? v[1] : v[i] for v in values]
        shared[i] = sum(selected)
    end
    for entry in endpoints
        scores[entry.id] ~ endpoint(entry, shared)
    end
end

@stestset "Later keyed components consume one parent computation" begin
    records = [(; id=:b, series=true, grid=[.5, 1.2, -.3], level=0.),
               (; id=:a, series=false, grid=Float64[], level=.4)]
    endpoints = [(; id=:right, weight=.1, y=[.2, .4, -.1]),
                 (; id=:left, weight=.2, y=[-.2, .3, .1])]
    brmi = pipeline((; records, endpoints, subjects=[1, 2, 3]))
    rk, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(brmi, "collection-shared-pipeline"; mod=@__MODULE__)
    names = BridgeStan.param_unc_names(stan.model)
    @test length(names) == 4
    descriptor = brm_descriptor(sb)
    position(collection, key) = only(brm_output_coordinates(descriptor, :bias, names;
        submodel=(collection, key)))
    transport = brm_coordinate_transport(rk, sb, names)
    shared(u) = [.9, 1.6, .1] .+
        u[position(:channels, :a)] .+ u[position(:channels, :b)]
    location(u, entry) = entry.weight .* shared(u) .+ u[position(:scores, entry.id)]
    oracle(u) = sum(logpdf.(Normal(), u)) +
        sum(sum(logpdf.(Normal.(location(u, entry), 1), entry.y)) for entry in endpoints)
    for point in (zeros(4), [.2, -.3, .1, -.2])
        expected(u) = oracle(brm_rk_point_to_stan(transport, u))
        value, gradient = check_consumer_point(problem, point, expected)
        stan_point = brm_rk_point_to_stan(transport, point)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model,
            stan_point; propto=false)
        @test value ≈ brm_stan_logdensity_to_rk(transport, point, stan_value)
        @test gradient ≈ brm_stan_gradient_to_rk(transport, point, stan_gradient)
        @test only(rk_draw_values(rk, reshape(point, :, 1), (:shared,)).shared) ≈ shared(stan_point)
        for entry in endpoints
            draws = rk_draw_values(rk, reshape(point, :, 1), (:loc,);
                submodel=(:scores, entry.id))
            @test only(draws.loc) ≈ location(stan_point, entry)
        end
    end
end
end
