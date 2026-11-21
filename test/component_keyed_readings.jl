module KeyedReadingsTest
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernelsDistributionKernels

@brm function sensor(record, signal)
    offset ~ Normal(0, 1)
    loc = signal[record.picks] .+ offset
    total = sum(loc)
    return (; total, loc)
end
const model = @brm begin
    gain ~ Normal(0, 1)
    @plate for i in eachindex(x)
        signal[i] = gain * x[i]
    end
    for record in records
        sensors[record.id] ~ sensor(record, signal)
    end
    y ~ Normal(sensors.total, 1)
end

@stestset "Read one component's scalar and variable-length returns by key" begin
    data = (; x=[2., -3., .75], records=[(; id=:b, picks=[1, 3]), (; id=:a, picks=[2])], y=[.3, .1])
    changed = (; data..., records=[(; id=:a, picks=Int[]), (; id=:b, picks=[2, 1, 3])], y=reverse(data.y))
    trained = model(data)
    rk, _ = consumer_problem(trained)
    sb = SBBRMI(trained; mod=@__MODULE__, total_groups=())
    U = [.5 -.25; .1 -.4; .25 .7] # gain, then offsets in key order
    @test_throws ArgumentError rk_draw_values(rk, U, (:loc,); submodel=(:sensors, :missing))
    @test_throws ArgumentError rk_draw_values(rk, U, (:unreturned,); submodel=(:sensors, :a))
    path = joinpath(mkpath(joinpath(tempdir(), "brm-rk-consumer")), "keyed-reading-returns.stan")
    for (design, r, s) in ((data, rk, sb), (changed, reprocess(rk, changed), reprocess(sb, changed)))
        stan = BRM.stan_instantiate(s; path).model
        names = BridgeStan.param_names(stan; include_tp=true)
        descriptor = brm_descriptor(s)
        problem = rk_logdensity_problem(r; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        draws = Dict(rec.id => rk_draw_values(r, U, (:total, :loc); submodel=(:sensors, rec.id))
            for rec in design.records)
        reference(rec, u) = u[1] .* design.x[rec.picks] .+ u[rec.id === :a ? 2 : 3]
        oracle(u) = sum(logpdf.(Normal(), u)) + sum(logpdf(Normal(sum(reference(rec, u)), 1), y)
            for (rec, y) in zip(design.records, design.y))
        for (j, u) in enumerate(eachcol(U))
            point = collect(u)
            value, gradient = check_consumer_point(problem, point, oracle)
            stan_value, stan_gradient = BridgeStan.log_density_gradient(stan, point; propto=false)
            @test stan_value ≈ value
            @test stan_gradient ≈ gradient
            constrained = BridgeStan.param_constrain(stan, point; include_tp=true)
            for record in design.records
                submodel = (:sensors, record.id)
                expected = reference(record, u)
                @test draws[record.id].total[j] ≈ sum(expected)
                @test draws[record.id].loc[j] ≈ expected
                indices = brm_output_coordinates(descriptor, :loc, names; submodel)
                @test constrained[indices] ≈ expected
            end
        end
    end
end

@brm function observed_sensor(y, picks, signal, heavy)
    offset ~ Normal(0, 0.6)
    if heavy
        @plate for s in eachindex(y)
            loc[s] = signal[s][picks[s]] .+ offset
            y[s] .~ student_t.(6, loc[s], 1)
        end
    else
        @plate for s in eachindex(y)
            loc[s] = signal[s][picks[s]] .+ offset
            y[s] .~ Normal.(loc[s], 1)
        end
    end
    return (; loc)
end
const single = @brm begin
    gain ~ Normal(0, 1)
    @plate for s in eachindex(x)
        signal[s] = gain .* x[s]
    end
    device ~ observed_sensor(y, picks, signal, heavy)
end

@stestset "A component observes readings at the supplied subject trajectories" begin
    x = [[2.5, 1.], [.2, 1.1, .8]]
    for (heavy, picks, y) in ((false, [[1, 2], Int[]], [[.9, .5], Float64[]]),
                              (true, [Int[], [3]], [Float64[], [.3]]))
        rk, problem = consumer_problem(single((; x, picks, y, heavy)))
        @test LogDensityProblems.dimension(problem) == 2
        family = heavy ? TDist(6) : Normal()
        function oracle(u)
            gain, offset = u
            value = logpdf(Normal(), gain) + logpdf(Normal(0, .6), offset)
            for s in eachindex(y), (row, measurement) in zip(picks[s], y[s])
                value += logpdf(family, measurement - (gain * x[s][row] + offset))
            end
            value
        end
        for u in ([.3, .4], [-.2, .15])
            check_consumer_point(problem, u, oracle)
        end
    end
end
end
