module ComponentLengthScaleFloorTest
# `length_scale_floor` in a record component's own `length_scale(...)` prior
# names each member's HSGP validity floor, as at model scope, and a member
# addresses its term the way its formula spells it, `hsgp(config.time)`, with
# or without a declaration guard. Each member is compared with the same smooth
# fitted alone (compiled Stan density and gradient, and the floor read back by
# member), and native RK with compiled Stan on the whole model.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@brm function guarded_clock(config)
    if config.smooth
        trend ~ 0 + hsgp(config.time; k=config.k, c=1.5)
        length_scale(trend, hsgp(config.time)) ~ Uniform(length_scale_floor, config.span)
        sd(trend, hsgp(config.time)) ~ Normal(0, config.amplitude)
        curve = trend
    else
        curve = config.baseline
    end
    @observe s walk member begin
        config.y[s] ~ Normal(walk .+ member[config.pick[s]], 1)
    end
    return curve
end

# The same smooth with its prior addresses outside any guard.
@brm function plain_clock(config)
    trend ~ 0 + hsgp(config.time; k=config.k, c=1.5)
    length_scale(trend, hsgp(config.time)) ~
        truncated(LogNormal(1, 1); lower=length_scale_floor, upper=config.span)
    sd(trend, hsgp(config.time)) ~ Normal(0, config.amplitude)
    @observe s walk member begin
        config.y[s] ~ Normal(walk .+ member[config.pick[s]], 1)
    end
    return trend
end

clocks(component) = brm(quote
    for c in sensors
        channels[c.id] ~ $component(c)
    end
    @plate for s in eachindex(input)
        shared[s] = 0.5 * input[s]
        observe(channels, s, shared[s])
    end
end; mod=@__MODULE__)

const one_guarded = @brm begin
    trend ~ 0 + hsgp(time; k=k, c=1.5)
    length_scale(trend, hsgp(time)) ~ Uniform(length_scale_floor, span)
    sd(trend, hsgp(time)) ~ Normal(0, amplitude)
    y ~ Normal(walk + trend[pick], 1)
end
const one_plain = @brm begin
    trend ~ 0 + hsgp(time; k=k, c=1.5)
    length_scale(trend, hsgp(time)) ~
        truncated(LogNormal(1, 1); lower=length_scale_floor, upper=span)
    sd(trend, hsgp(time)) ~ Normal(0, amplitude)
    y ~ Normal(walk + trend[pick], 1)
end

const input = [2.0, -1.0]
# Smooth members on time axes of different widths, so their floors differ.
const a = (; id=:a, smooth=true, time=[0.1, 0.35, 0.75], k=3, span=4.0, amplitude=0.8,
    pick=[[2, 3], Int[]], y=[[0.6, 0.8], Float64[]])
const b = (; id=:b, smooth=false, baseline=[0.25, 0.13], pick=[Int[], [2]], y=[Float64[], [0.7]])
const c = (; id=:c, smooth=true, time=[-4.0, 0.0, 4.0, 5.2], k=4, span=30.0, amplitude=1.3,
    pick=[[1], [2]], y=[[0.3], [0.2]])
const NATIVE = (; sd=:sigma, length_scale=:rho_iso, basis_weights=:beta_raw)

# `smooth` lists the members with a smooth in key order: their position in
# the RK variant's carriers.
function check_members(component, single, sensors, smooth, variant, label)
    m = clocks(component)((; sensors, input))
    desc = brm_descriptor(SBBRMI(m; mod=@__MODULE__, total_groups=()))
    stan = consumer_stan(m, label; mod=@__MODULE__).model
    names = BridgeStan.param_unc_names(stan)
    constrained = BridgeStan.param_names(stan)
    rk, problem = consumer_problem(m)
    rknames = coordinate_names(rk.model.layout)
    stan_to_rk = zeros(Int, length(names))
    floors = Float64[]
    refs = map(enumerate(smooth)) do (position, record)
        address = (:channels, record.id)
        walk = reduce(vcat, [fill(0.5 * input[s], length(record.y[s])) for s in eachindex(input)])
        source = single((; record.time, record.k, record.span, record.amplitude, walk,
            pick=reduce(vcat, record.pick), y=reduce(vcat, record.y)))
        ref = consumer_stan(source, label * "-single"; mod=@__MODULE__).model
        rdesc = brm_descriptor(SBBRMI(source; mod=@__MODULE__, total_groups=()))
        # The member's floor is its own term's floor, read back by member.
        floor = brm_length_scale_floor(desc, :trend; term=:hsgp_config_time, submodel=address)
        @test floor == brm_length_scale_floor(rdesc, :trend; term=:hsgp_time)
        push!(floors, floor)
        # Its length scale is declared above that floor: the lowest
        # representable unconstrained value constrains to it.
        scale = only(brm_term_coordinates(desc, :trend, names; parameter=:length_scale,
            submodel=address).coordinates)
        low = zeros(length(names))
        low[scale] = -40.0
        @test BridgeStan.param_constrain(stan, low)[only(brm_term_coordinates(desc, :trend,
            constrained; parameter=:length_scale, submodel=address).coordinates)] ≈ floor rtol=1e-12
        rnames = BridgeStan.param_unc_names(ref)
        mapping = zeros(Int, length(rnames))
        for parameter in (:sd, :length_scale, :basis_weights)
            src = brm_term_coordinates(rdesc, :trend, rnames; parameter)
            dst = brm_term_coordinates(desc, :trend, names; parameter, submodel=address)
            mapping[src.coordinates] = dst.coordinates
            for (j, index) in enumerate(dst.coordinates)
                suffix = parameter === :basis_weights ? ".$position.$j" : ".$position"
                stan_to_rk[index] = coordinate(rknames,
                    Symbol("channels_member_", variant, ".hsgp_config_time.",
                        NATIVE[parameter], suffix))
            end
        end
        (; ref, mapping)
    end
    @test allunique(floors)
    n = length(names)
    @test sort(reduce(vcat, r.mapping for r in refs)) == 1:n
    @test sort(stan_to_rk) == 1:n
    for u in (zeros(n), collect(range(-0.37, 0.31; length=n)))
        value, gradient = BridgeStan.log_density_gradient(stan, u; propto=false)
        native_value, native_gradient =
            LogDensityProblems.logdensity_and_gradient(problem, u[invperm(stan_to_rk)])
        @test native_value ≈ value
        @test native_gradient[stan_to_rk] ≈ gradient
        expected = 0.0
        for record in sensors
            record.smooth && continue
            expected += logpdf(Normal(0.5 * input[2] + record.baseline[2], 1), only(record.y[2]))
        end
        for (; ref, mapping) in refs
            part, grad = BridgeStan.log_density_gradient(ref, u[mapping]; propto=false)
            expected += part
            @test gradient[mapping] ≈ grad
        end
        @test value ≈ expected
    end
end

@stestset "A guarded member's length_scale_floor is its own term's floor" begin
    check_members(guarded_clock, one_guarded, [c, b, a], (a, c), 1, "member-floor-guarded")
end

@stestset "An unguarded member addresses hsgp(config.time) and names its floor" begin
    e = (; id=:e, smooth=true, time=[1.0, 2.0, 8.0], k=3, span=40.0, amplitude=0.5,
        pick=[Int[], [1, 3]], y=[Float64[], [0.1, -0.2]])
    check_members(plain_clock, one_plain, [c, e, a], (a, c, e), :all, "member-floor-plain")
end

# A member whose intercept is a margin of the parent's correlated block reads
# a port only the parent's call binds, so its own program is never traced
# alone; its floor is read without tracing it.
@brm function margin_clock(config)
    base ~ 1 + (1 | parentblock(:p) | config.unit)
    trend ~ 0 + hsgp(config.time; k=config.k, c=1.5)
    length_scale(trend, hsgp(config.time)) ~ Uniform(length_scale_floor, config.span)
    @observe s walk member begin
        config.y[s] ~ Normal(walk .+ member.base[config.unit_rows[s]] .+
            member.curve[config.pick[s]], 1)
    end
    return (; base, curve=trend)
end

@stestset "A member beside a parent-block margin reads back its floor" begin
    margined = @brm begin
        root ~ 0 + (1 | p | unit)
        primary ~ Normal(root, 1)
        for c in sensors
            channels[c.id] ~ margin_clock(c)
        end
        @plate for s in eachindex(input)
            shared[s] = 0.5 * input[s]
            observe(channels, s, shared[s])
        end
    end
    member(record) = (; record.id, record.time, record.k, record.span, unit=[1, 2],
        unit_rows=[fill(1, length(record.pick[1])), fill(2, length(record.pick[2]))],
        record.pick, record.y)
    m = margined((; unit=[1, 2], primary=[0.2, -0.1], sensors=[member(c), member(a)], input))
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    @test stan_code(sb) isa AbstractString
    @test RKBRMI(m) isa RKBRMI
    desc = brm_descriptor(sb)
    for record in (a, c)
        alone = one_guarded((; record.time, record.k, record.span, record.amplitude,
            walk=zeros(3), pick=[1, 1, 1], y=zeros(3)))
        @test brm_length_scale_floor(desc, :trend; term=:hsgp_config_time,
                submodel=(:channels, record.id)) ==
            brm_length_scale_floor(brm_descriptor(SBBRMI(alone; mod=@__MODULE__,
                total_groups=())), :trend; term=:hsgp_time)
    end
    @test_throws "available term labels are (:hsgp_config_time,)" brm_length_scale_floor(
        desc, :trend; term=:hsgp_time, submodel=(:channels, :a))
end

@stestset "length_scale_floor outside a member's length_scale prior is still refused" begin
    @brm function misread(config)
        shift ~ Normal(length_scale_floor, 1)
        @observe s walk member begin
            config.y[s] ~ Normal(walk .+ member, 1)
        end
        return shift
    end
    @brm function misplaced(config)
        trend ~ 0 + hsgp(config.time; k=3, c=1.5)
        sd(trend, hsgp(config.time)) ~ Uniform(length_scale_floor, 2.0)
        @observe s walk member begin
            config.y[s] ~ Normal(walk .+ member[config.pick[s]], 1)
        end
        return trend
    end
    for (component, message) in ((misread, "reads undeclared input `length_scale_floor`"),
                                 (misplaced, "is only valid in its own `length_scale(...)` prior"))
        m = clocks(component)((; sensors=[a], input))
        @test_throws message SBBRMI(m; mod=@__MODULE__, total_groups=())
        @test_throws message RKBRMI(m)
    end
    # At model scope a record-field address names no term of the formula.
    model = @brm begin
        mu ~ 0 + hsgp(x; k=4, c=1.5)
        length_scale(mu, hsgp(record.x)) ~ Uniform(length_scale_floor, 10.0)
        y ~ Normal(mu, 1)
    end
    @test_throws "matches no `hsgp(record.x)` term" SBBRMI(
        model((; x=collect(1.0:5.0), y=zeros(5))); mod=@__MODULE__)
end

end
