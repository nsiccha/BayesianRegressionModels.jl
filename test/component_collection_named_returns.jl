module CollectionNamedReturnsTest
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels

# A consumer project may refuse a prepared program that dispatches at run time
# (the ReactiveKernels preference `type_stability = "error"`).
strict(f) = ReactiveKernels._with_type_stability(f, :error, :all)

# A curve on the member's own grid beside a fitted scale.
@brm function scaled_reading(config)
    gain ~ Normal(0, 1)
    scale ~ LogNormal(-0.7, 1.0)
    curve = gain .* config.x
    @observe s walk member begin
        config.y[s] ~ Normal(walk .+ member.curve[config.pick[s]], member.scale)
    end
    return (; curve, scale)
end
const scaled_model = @brm begin
    for c in sensors
        channels[c.id] ~ scaled_reading(c)
    end
    @plate for s in eachindex(input)
        shared[s] = 0.5 * input[s]
        observe(channels, s, shared[s])
    end
end

# A per-site vector in the parent's correlated block beside two scalars.
@brm function site_reading(entry)
    offset ~ Normal(0.0, 1.0)
    shift ~ 0 + (1 | parentblock(:p) | entry.site)
    scale ~ LogNormal(-0.7, 1.0)
    @observe s trajectory member begin
        entry.y[s] ~ Normal(member.offset .+ member.shift[entry.rows[s]] .+
            trajectory[entry.stack[s]], member.scale)
    end
    return (; offset, shift, scale)
end
const site_model = @brm begin
    level ~ 1 + (1 | p | site)
    effect(level, Intercept) ~ Normal(0.0, 1.0)
    sd(:, p) ~ Exponential(0.3333333333333333)
    cor(:, p) ~ LKJCholesky(n_margins, 2.0)
    for entry in records
        channels[entry.id] ~ site_reading(entry)
    end
    @plate for i in eachindex(level)
        path = exp(level[i]) .* exp.(-0.1 .* read_time[i])
        curve[i] = path
        observe(channels, i, curve[i])
    end
end
# The same model without components.
const site_twin = @brm begin
    level ~ 1 + (1 | p | site)
    effect(level, Intercept) ~ Normal(0.0, 1.0)
    sd(:, p) ~ Exponential(0.3333333333333333)
    cor(:, p) ~ LKJCholesky(n_margins, 2.0)
    a_offset ~ Normal(0.0, 1.0)
    a_shift ~ 0 + (1 | p | a_site)
    a_scale ~ LogNormal(-0.7, 1.0)
    b_offset ~ Normal(0.0, 1.0)
    b_shift ~ 0 + (1 | p | b_site)
    b_scale ~ LogNormal(-0.7, 1.0)
    ya ~ Normal(a_offset + a_shift[a_rows] + exp(level[a_rows]) * a_decay, a_scale)
    yb ~ Normal(b_offset + b_shift[b_rows] + exp(level[b_rows]) * b_decay, b_scale)
end

@stestset "A named return reads a member's curve beside its fitted scale" begin
    sensors = [(; id=:b, x=[2., 3.], pick=[[1], [2]], y=[[.3], [.2]]),
               (; id=:a, x=[4., 7., 8.], pick=[[2, 3], Int[]], y=[[.6, .8], Float64[]])]
    input = [3., 1.]
    m = scaled_model((; sensors, input))
    rk, problem = strict(() -> consumer_problem(m))
    names = coordinate_names(rk.model.layout)
    # Members are fitted in key order: a, then b.
    members = reverse(sensors)
    gain = [coordinate(names, Symbol("channels_member_all.gain.", i)) for i in 1:2]
    scale = [coordinate(names, Symbol("channels_member_all.scale.", i)) for i in 1:2]
    function oracle(u)
        value = zero(eltype(u))
        for (i, sensor) in enumerate(members)
            g, log_s = u[gain[i]], u[scale[i]]
            s = exp(log_s)
            value += logpdf(Normal(), g) + logpdf(LogNormal(-0.7, 1.0), s) + log_s
            for subject in eachindex(input), (row, y) in zip(sensor.pick[subject], sensor.y[subject])
                value += logpdf(Normal(0.5 * input[subject] + g * sensor.x[row], s), y)
            end
        end
        value
    end
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, "named-curve-and-scale"; mod=@__MODULE__)
    desc = brm_descriptor(sb)
    snames = BridgeStan.param_unc_names(stan.model)
    # Stan position of each RK coordinate, by the member's keyed address.
    stan_at = zeros(Int, 4)
    for (i, sensor) in enumerate(members), (field, at) in ((:gain, gain), (:scale, scale))
        stan_at[at[i]] = only(brm_output_coordinates(desc, field, snames;
            submodel=(:channels, sensor.id)))
    end
    @test sort(stan_at) == 1:4
    for u in (zeros(4), [0.14, -0.27, 0.31, -0.22], [-0.31, 0.21, -0.4, 0.17])
        value, gradient = check_consumer_point(problem, u, oracle)
        stan_u = similar(u); stan_u[stan_at] = u
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_u; propto=false)
        @test stan_value ≈ value
        @test stan_gradient[stan_at] ≈ gradient
    end
end

@stestset "A named return reads a parent-block vector beside two scalars" begin
    reads = [[1, 2, 3], [1, 3], [2, 3], [1, 2, 3]]
    entry(id, level) = (; id, stack=reads, y=[level .+ 0.1 .* r for r in reads],
        site=collect(1:4), rows=[fill(s, length(reads[s])) for s in eachindex(reads)])
    records = [entry(:ast, -0.3), entry(:alt, 0.2)]   # supplied out of key order
    read_time = [[2.0, 12.0, 48.0], [1.0, 9.0, 30.0], [3.0, 14.0, 50.0], [2.5, 11.0, 40.0]]
    m = site_model((; site=collect(1:4), read_time, records, n_margins=3))
    rk, problem = strict(() -> consumer_problem(m))
    flat(record) = (; rows=reduce(vcat, record.rows), y=reduce(vcat, record.y),
        decay=reduce(vcat, [exp.(-0.1 .* read_time[s][record.stack[s]]) for s in eachindex(reads)]))
    a, b = flat(records[2]), flat(records[1])        # a = :alt, b = :ast
    twin = site_twin((; site=collect(1:4), n_margins=3, a_site=collect(1:4), b_site=collect(1:4),
        a_rows=a.rows, a_decay=a.decay, ya=a.y, b_rows=b.rows, b_decay=b.decay, yb=b.y))
    rkt, reference = consumer_problem(twin)
    names = coordinate_names(rk.model.layout)
    tnames = coordinate_names(rkt.model.layout)
    rename = Dict(Symbol("channels_member_all.offset.1") => :a_offset,
        Symbol("channels_member_all.offset.2") => :b_offset,
        Symbol("channels_member_all.scale.1") => :a_scale,
        Symbol("channels_member_all.scale.2") => :b_scale)
    to_twin = [coordinate(tnames, get(rename, name, name)) for name in names]
    @test sort(to_twin) == 1:length(tnames) == 1:23
    n = length(names)
    points = (zeros(n), collect(range(-0.37, 0.31; length=n)), [0.3sin(1.7i) for i in 1:n])
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    stan = consumer_stan(m, "named-parent-block-vector"; mod=@__MODULE__)
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    for u in points
        v = similar(u); v[to_twin] = u
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        expected, reference_gradient = LogDensityProblems.logdensity_and_gradient(reference, v)
        @test value ≈ expected atol=2e-11 rtol=2e-11
        @test gradient ≈ reference_gradient[to_twin] atol=2e-9 rtol=2e-9
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(stan.model, stan_u; propto=false)
        @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value)
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient)
    end
end
end
