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

@brm function sensor_range(setting)
    intercept ~ Normal(0, 1)
    residual ~ Exponential(1)
    if setting.limited
        @observe s trajectory sensor begin
            setting.y[s] ~ censored(Normal(trajectory .+ sensor.location, sensor.scale);
                lower=setting.limit[s])
        end
    else
        @observe s trajectory sensor begin
            setting.y[s] ~ Normal(trajectory .+ sensor.location, sensor.scale)
        end
    end
    return (; location=intercept, scale=residual)
end

const ranged = @brm begin
    for config in settings
        sensors[config.id] ~ sensor_range(config)
    end
    @plate for s in eachindex(input)
        shared[s] = 0.5 * input[s]
        observe(sensors, s, shared[s])
    end
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

@brm function vector_reading(config)
    gain ~ Normal(0,1)
    curve = gain .* config.x
    @observe s walk member begin
        config.y[s] ~ Normal(walk .+ member[config.pick[s]], 1)
    end
    return curve
end
const vector_model = @brm begin
    for c in sensors
        channels[c.id] ~ vector_reading(c)
    end
    @plate for s in eachindex(input)
        shared[s] = 0.5 * input[s]
        observe(channels, s, shared[s])
    end
end

@stestset "Subject observations index each member's own vector" begin
    sensors = [(;id=:b,x=[2.,3.],pick=[[1],[2]],y=[[.3],[.2]]),
               (;id=:a,x=[4.,7.,8.],pick=[[2,3],Int[]],y=[[.6,.8],Float64[]])]
    input = [3.,1.]
    m = vector_model((;sensors,input))
    _, problem = consumer_problem(m)
    stan = consumer_stan(m, "vector-member-observations"; mod=@__MODULE__)
    function oracle(u)
        value = sum(x -> logpdf(Normal(),x), u)
        for (gain, sensor) in zip(u, reverse(sensors)), s in eachindex(input)
            for (row, y) in zip(sensor.pick[s], sensor.y[s])
                value += logpdf(Normal(0.5*input[s] + gain*sensor.x[row],1), y)
            end
        end
        value
    end
    for u in (zeros(2), [.14,-.27], [-.31,.21])
        value, grad = check_consumer_point(problem, u, oracle)
        svalue, sgrad = BridgeStan.log_density_gradient(stan.model, u; propto=false)
        @test svalue ≈ value
        @test sgrad ≈ grad
    end
end

@brm function sensor_clock(config)
    if config.smooth
        trend ~ 0 + hsgp(config.time; k=config.k, c=1.5)
        curve = trend
    else
        curve = config.baseline
    end
    @observe s walk member begin
        config.y[s] ~ Normal(walk .+ member[config.pick[s]], 1)
    end
    return curve
end
const clocks = @brm begin
    for c in sensors
        channels[c.id] ~ sensor_clock(c)
    end
    @plate for s in eachindex(input)
        shared[s] = 0.5 * input[s]
        observe(channels, s, shared[s])
    end
end
const one_clock = @brm begin
    trend ~ 0 + hsgp(time; k=k, c=1.5)
    y ~ Normal(walk + trend[pick], 1)
end

@stestset "Per-member time smooths enter the shared subject likelihood" begin
    input = [2.,-1.]
    a = (;id=:a,smooth=true,time=[.1,.35,.75],k=3,pick=[[2,3],Int[]],y=[[.6,.8],Float64[]])
    b = (;id=:b,smooth=false,baseline=[.25,.13],pick=[Int[],[2]],y=[Float64[],[.7]])
    c = (;id=:c,smooth=true,time=[-1.,0.,1.,1.3],k=4,pick=[[1],[2]],y=[[.3],[.2]])
    m = clocks((;sensors=[c,b,a],input))
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    desc = brm_descriptor(sb)
    stan = consumer_stan(m, "observed-member-clocks"; mod=@__MODULE__).model
    names = BridgeStan.param_unc_names(stan)
    rk, problem = consumer_problem(m)
    rknames = ReactiveKernelsPPL.coordinate_names(rk.model.layout)
    stan_to_rk = zeros(Int,length(names))
    refs = map((a,c), (1,2)) do record, position
        walk = reduce(vcat, [fill(0.5*input[s],length(record.y[s])) for s in eachindex(input)])
        source = one_clock((;record.time,record.k,walk,pick=reduce(vcat,record.pick),y=reduce(vcat,record.y)))
        ref = consumer_stan(source, "observed-single-clock"; mod=@__MODULE__).model
        rdesc = brm_descriptor(SBBRMI(source; mod=@__MODULE__, total_groups=()))
        rnames = BridgeStan.param_unc_names(ref)
        mapping = zeros(Int,length(rnames))
        for parameter in (:sd,:length_scale,:basis_weights)
            src = brm_term_coordinates(rdesc, :trend, rnames; parameter)
            dst = brm_term_coordinates(desc, :trend, names; parameter, submodel=(:channels,record.id))
            mapping[src.coordinates] = dst.coordinates
            native = parameter === :sd ? :sigma : parameter === :length_scale ? :rho_iso : :beta_raw
            for (j, index) in enumerate(dst.coordinates)
                suffix = parameter === :basis_weights ? ".$position.$j" : ".$position"
                stan_to_rk[index] = coordinate(rknames, Symbol("channels_member_1.hsgp_config_time.",native,suffix))
            end
        end
        (;ref,mapping)
    end
    n = length(names)
    @test sort(reduce(vcat, r.mapping for r in refs)) == 1:n
    @test sort(stan_to_rk) == 1:n
    for u in (zeros(n), collect(range(-.37,.31;length=n)))
        value, gradient = BridgeStan.log_density_gradient(stan, u; propto=false)
        native_value, native_gradient = LogDensityProblems.logdensity_and_gradient(problem,u[invperm(stan_to_rk)])
        @test native_value ≈ value
        @test native_gradient[stan_to_rk] ≈ gradient
        expected = logpdf(Normal(0.5*input[2]+b.baseline[2],1), only(b.y[2]))
        for (;ref,mapping) in refs
            part, grad = BridgeStan.log_density_gradient(ref, u[mapping]; propto=false)
            expected += part
            @test gradient[mapping] ≈ grad
        end
        @test value ≈ expected
    end
end

@stestset "Mixed observation families read one trajectory per subject" begin
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

@stestset "Censored sensor rows use their fitted scale and shared trajectory" begin
    settings = [(id=:a, limited=false, y=[[0.3], [0.2,0.1]]),
                (id=:b, limited=true, y=[[0.2,0.6], Float64[]],
                 limit=[[0.2,0.2], Float64[]])]
    m = ranged((; settings, input=[1.,2.]))
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, "repeated-censored-sensors"; mod=@__MODULE__)
    rk, problem = consumer_problem(m)
    @test rk.model.layout.total == 4
    desc = brm_descriptor(sb)
    snames = BridgeStan.param_unc_names(stan.model)
    rnames = ReactiveKernelsPPL.coordinate_names(rk.model.layout)
    function indices(i, field)
        address = (:sensors, settings[i].id)
        s = only(brm_output_coordinates(desc, field, snames; submodel=address))
        r = coordinate(rnames, Symbol("sensors_member_", Int(settings[i].limited), ".", field, ".1"))
        (; s, r)
    end
    offsets = [indices(i, :intercept) for i in eachindex(settings)]
    scales = [indices(i, :residual) for i in eachindex(settings)]
    function oracle(u)
        value = zero(eltype(u))
        for (i, record) in enumerate(settings)
            bias = u[offsets[i].r]
            log_scale = u[scales[i].r]
            scale = exp(log_scale)
            value += logpdf(Normal(), bias) + logpdf(Exponential(), scale) + log_scale
            for s in eachindex(record.y), (row, y) in enumerate(record.y[s])
                law = Normal(0.5*s + bias, scale)
                value += record.limited && y == record.limit[s][row] ? logcdf(law, y) : logpdf(law, y)
            end
        end
        value
    end
    perm = zeros(Int, 4)
    for index in vcat(offsets, scales)
        perm[index.s] = index.r
    end
    for u in (zeros(4), [-0.23, 0.12, 0.38, -0.27])
        value, grad = check_consumer_point(problem, u, oracle)
        svalue, sgrad = BridgeStan.log_density_gradient(stan.model, u[perm]; propto=false)
        @test value ≈ svalue
        @test grad[perm] ≈ sgrad
    end
end
end
