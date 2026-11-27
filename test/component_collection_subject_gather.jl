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
    n_members = constant(length(records))
    @plate for i in eachindex(y)
        selected = [brm_subject_value(values[j], i) for j in 1:n_members]
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
    @test rk.model.layout.total == 2
    transport = brm_coordinate_transport(rk, sb, names)
    for u in (zeros(2), [.2, -.3])
        expected(v) = oracle(brm_rk_point_to_stan(transport, v))
        value, gradient = check_consumer_point(problem, u, expected)
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_u; propto=false)
        @test stan_value ≈ oracle(stan_u)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value)
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient)
        constrained_names = BridgeStan.param_names(stan.model; include_tp=true)
        constrained = BridgeStan.param_constrain(stan.model, stan_u; include_tp=true)
        draws = rk_draw_values(rk, reshape(u, :, 1), (:values, :shared))
        @test only(draws.shared) ≈ means(stan_u)
        for (j, record) in enumerate(data.records)
            expected_value = stan_u[indices[record.id]] .+
                (record.series ? record.grid : record.level)
            @test only(draws.values)[j] ≈ expected_value
            member = rk_draw_values(rk, reshape(u, :, 1), (:value,);
                submodel=(:channels, record.id))
            @test only(member.value) ≈ expected_value
            positions = brm_output_coordinates(descriptor, :value, constrained_names;
                submodel=(:channels, record.id))
            @test constrained[positions] ≈ vec([expected_value;])
        end
    end
    # Reorder keys and change the subject count without changing fitted coordinates.
    changed = (; data..., y=[.1, -.4, .2, .3], weights=[.7, -.2],
        records=[(; record..., grid=record.series ? [.4, .7, -.1, .8] : Float64[])
            for record in reverse(data.records)])
    replay = reprocess(rk, changed)
    replay_sb = reprocess(sb, changed)
    replay_stan = BRM.stan_instantiate(replay_sb;
        path=joinpath(tempdir(), "brm-rk-consumer", "collection-replay-" * label * ".stan"))
    @test BridgeStan.param_unc_names(replay_stan.model) == names
    @test coordinate_names(replay.model.layout) == coordinate_names(rk.model.layout)
    point = [.2, -.3]
    replay_problem = rk_logdensity_problem(replay; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = check_consumer_point(replay_problem, point,
        u -> oracle(brm_rk_point_to_stan(transport, u), changed))
    stan_point = brm_rk_point_to_stan(transport, point)
    stan_value, stan_gradient = BridgeStan.log_density_gradient(replay_stan.model,
        stan_point; propto=false)
    @test value ≈ brm_stan_logdensity_to_rk(transport, point, stan_value)
    @test gradient ≈ brm_stan_gradient_to_rk(transport, point, stan_gradient)
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

@stestset "Subject gathering prepares under strict type stability" begin
    vector_b = (; id=:b, series=true, grid=[.5, 1.2, -.3], level=0.)
    vector_a = (; id=:a, series=true, grid=[1.1, -.2, .7], level=0.)
    scalar_b = (; id=:b, series=false, grid=Float64[], level=.4)
    scalar_a = (; id=:a, series=false, grid=Float64[], level=-.3)
    # A heterogeneous collection deliberately exposes a Vector{Any} on RK;
    # its density and gradients are checked above under the normal setting.
    for (label, records) in (("scalars", [scalar_b, scalar_a]),
            ("vectors", [vector_b, vector_a]))
        @testset "$label" begin
            ReactiveKernels._with_type_stability(:error, :all) do
                brmi = gathered((; records, weights=[.3, .8], y=[.2, -.1, .3]))
                rk, problem = consumer_problem(brmi)
                @test rk.model.layout.total == 2
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, [.2, -.3])
                @test isfinite(value)
                @test all(isfinite, gradient)
            end
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
    n_members = constant(length(records))
    @plate for i in eachindex(subjects)
        selected = [brm_subject_value(values[j], i) for j in 1:n_members]
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
    data = (; records, endpoints, subjects=[1, 2, 3])
    brmi = pipeline(data)
    rk, problem = consumer_problem(brmi)
    @test rk.model.layout.total == 4
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(brmi, "collection-shared-pipeline"; mod=@__MODULE__)
    names = BridgeStan.param_unc_names(stan.model)
    @test length(names) == 4
    descriptor = brm_descriptor(sb)
    position(collection, key) = only(brm_output_coordinates(descriptor, :bias, names;
        submodel=(collection, key)))
    transport = brm_coordinate_transport(rk, sb, names)
    function shared(u, design=data)
        result = zeros(length(design.subjects))
        for record in design.records
            result .+= u[position(:channels, record.id)] .+
                (record.series ? record.grid : record.level)
        end
        result
    end
    location(u, entry, design=data) =
        entry.weight .* shared(u, design) .+ u[position(:scores, entry.id)]
    oracle(u, design=data) = sum(logpdf.(Normal(), u)) +
        sum(sum(logpdf.(Normal.(location(u, entry, design), 1), entry.y))
            for entry in design.endpoints)
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
            positions = brm_output_coordinates(descriptor, :loc,
                BridgeStan.param_names(stan.model; include_tp=true, include_gq=true);
                submodel=(:scores, entry.id))
            @test BridgeStan.param_constrain(stan.model, stan_point;
                include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(stan.model, 1))[positions] ≈
                location(stan_point, entry)
        end
    end
    changed = (; subjects=[1, 2, 3, 4],
        records=[(; record..., grid=record.series ? [.4, .7, -.1, .8] : Float64[])
            for record in reverse(records)],
        endpoints=[(; entry..., weight=entry.weight + .3, y=[.1, -.4, .2, .3])
            for entry in reverse(endpoints)])
    replay = reprocess(rk, changed)
    @test coordinate_names(replay.model.layout) == coordinate_names(rk.model.layout)
    replay_stan = BRM.stan_instantiate(reprocess(sb, changed);
        path=joinpath(tempdir(), "brm-rk-consumer", "pipeline-replay.stan"))
    @test BridgeStan.param_unc_names(replay_stan.model) == names
    point = [.2, -.3, .1, -.2]
    replay_problem = rk_logdensity_problem(replay; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = check_consumer_point(replay_problem, point,
        u -> oracle(brm_rk_point_to_stan(transport, u), changed))
    stan_point = brm_rk_point_to_stan(transport, point)
    stan_value, stan_gradient = BridgeStan.log_density_gradient(replay_stan.model,
        stan_point; propto=false)
    @test value ≈ brm_stan_logdensity_to_rk(transport, point, stan_value)
    @test gradient ≈ brm_stan_gradient_to_rk(transport, point, stan_gradient)
    @test only(rk_draw_values(replay, reshape(point, :, 1), (:shared,)).shared) ≈
        shared(stan_point, changed)
    for entry in changed.endpoints
        draws = rk_draw_values(replay, reshape(point, :, 1), (:loc,);
            submodel=(:scores, entry.id))
        @test only(draws.loc) ≈ location(stan_point, entry, changed)
    end
end

@brm function shared_reading(entry)
    if entry.series
        value ~ 1 + (1 | parentblock(:p) | entry.subject)
        effect(value, Intercept) ~ Normal(entry.center, entry.scale)
    else
        value ~ Normal(entry.center, entry.scale)
    end
    return (; value)
end

const parent_pipeline = @brm begin
    root ~ 0 + (1 | p | subject)
    sd(:, p) = 0.4
    cor(:, p) = [1.0 0.3; 0.3 1.0]
    for entry in records
        channels[entry.id] ~ shared_reading(entry)
    end
    values = channels.value
    n_members = constant(length(records))
    @plate for i in eachindex(subjects)
        selected = [brm_subject_value(values[j], i) for j in 1:n_members]
        shared[i] = root[i] + sum(selected)
    end
    for entry in endpoints
        scores[entry.id] ~ endpoint(entry, shared)
    end
end

@stestset "Gathered member values retain the shared parent block" begin
    subject = ["first", "second", "third"]
    records = [(; id=:b, series=true, subject, center=.3, scale=.7),
               (; id=:a, series=false, subject, center=-.2, scale=.8)]
    endpoints = [(; id=:right, weight=.1, y=[.2, .4, -.1]),
                 (; id=:left, weight=.2, y=[-.2, .3, .1])]
    data = (; subject, subjects=collect(1:3), records, endpoints)
    brmi = parent_pipeline(data)
    rk, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test only(ranef_blocks(sb)).n_terms == 2
    stan = consumer_stan(brmi, "collection-parent-pipeline"; mod=@__MODULE__)
    names = BridgeStan.param_unc_names(stan.model)
    # Six standardised effects, two upstream locations and two downstream biases.
    @test length(names) == rk.model.layout.total == 10
    transport = brm_coordinate_transport(rk, sb, names)
    descriptor = brm_descriptor(sb)
    constrained_names = BridgeStan.param_names(stan.model; include_tp=true, include_gq=true)
    for point in (zeros(10), collect(range(-.3, .4; length=10)))
        stan_point = brm_rk_point_to_stan(transport, point)
        oracle(u) = brm_stan_logdensity_to_rk(transport, u,
            BridgeStan.log_density(stan.model, brm_rk_point_to_stan(transport, u);
                propto=false))
        value, gradient = check_consumer_point(problem, point, oracle)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model,
            stan_point; propto=false)
        @test value ≈ brm_stan_logdensity_to_rk(transport, point, stan_value)
        @test gradient ≈ brm_stan_gradient_to_rk(transport, point, stan_gradient)
        constrained = BridgeStan.param_constrain(stan.model, stan_point;
            include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(stan.model, 1))
        root = constrained[brm_output_coordinates(descriptor, :root, constrained_names)]
        expected = copy(root)
        for record in records
            member = constrained[brm_output_coordinates(descriptor, :value,
                constrained_names; submodel=(:channels, record.id))]
            expected .+= member
            drawn = only(rk_draw_values(rk, reshape(point, :, 1), (:value,);
                submodel=(:channels, record.id)).value)
            @test vec([drawn;]) ≈ member
        end
        @test only(rk_draw_values(rk, reshape(point, :, 1), (:shared,)).shared) ≈ expected
        for entry in endpoints
            bias = only(constrained[brm_output_coordinates(descriptor, :bias,
                constrained_names; submodel=(:scores, entry.id))])
            location = entry.weight .* expected .+ bias
            @test constrained[brm_output_coordinates(descriptor, :loc,
                constrained_names; submodel=(:scores, entry.id))] ≈ location
            @test only(rk_draw_values(rk, reshape(point, :, 1), (:loc,);
                submodel=(:scores, entry.id)).loc) ≈ location
        end
    end
end
end
