module CollectionObservationsTest
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels
@brm function reading(config)
    shift ~ Normal(0, config.scale)
    if config.robust
        @observe s trajectory member begin
            config.y[s] ~ student_t(5, trajectory .+ member, 1)
        end
    else
        @observe s trajectory member begin
            config.y[s] ~ Normal(trajectory .+ member, 1)
        end
    end
    return shift
end
const model = @brm begin
    for item in records
        channels[item.id] ~ reading(item)
    end
    @plate for s in eachindex(input)
        trajectory[s] = 0.5 * input[s]
        observe(channels, s, trajectory[s])
    end
end
const records=[(id=:b,robust=false,scale=.7,y=[[0.1],[0.2, 0.3]]),
               (id=:c,robust=true,scale=.9,y=[[0.6],Float64[]]),
               (id=:a,robust=false,scale=.5,y=[[0.4,0.3],[0.5]])]
const df = (; records, input=[1.,2.])

@testset "Mixed observation families read one trajectory per subject" begin
    @test brm_columns(model) == (:input, :records)
    m = model(df)
    _, problem = consumer_problem(m)
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    for fit in (sb, reprocess(sb, (; df..., records=reverse(records))))
        desc = brm_descriptor(fit)
        for (key, expected, n) in ((:a, [1,2,4], 6), (:b, [3,5,6], 6), (:c, [1], 1))
            output = brm_output(desc, :y; submodel=(:channels,key), role=:posterior_predictive)
            axis = [string(output.name, ".", i) for i in 1:n]
            @test brm_output_coordinates(output, axis) == expected
        end
    end
    stan = consumer_stan(m, "repeated-observation-families"; mod=@__MODULE__)
    # The Gaussian members are ordered a,b and the Student-t member is c.
    function oracle(u)
        value = sum(logpdf.(Normal.(0, [0.5, 0.7, 0.9]), u))
        for (i, record) in enumerate(records[[3, 1, 2]]), s in eachindex(df.input)
            loc = 0.5 * df.input[s] + u[i]
            law = record.robust ? LocationScale(loc, 1.0, TDist(5)) : Normal(loc, 1.0)
            value += sum(x -> logpdf(law, x), record.y[s]; init=0.0)
        end
        value
    end
    for u in (zeros(3), [-0.35, 0.12, 0.28], [0.7, -0.2, -0.43])
        value, gradient = check_consumer_point(problem, u, oracle)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, u;
            propto=false, jacobian=true)
        @test stan_value ≈ value
        @test stan_gradient ≈ gradient
    end
end
end
