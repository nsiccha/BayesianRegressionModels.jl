module MemberPopulationTest

# A member of a component repeated over records owns its own population
# coefficients. They are addressed by the member's key and paired across the
# two backends by meaning, checked against the model that writes each member
# out.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using ReactiveKernelsDistributionKernels

@brm function dosed(settings)
    shift ~ 1 + settings.dose
    effect(shift, Intercept) ~ Normal(0.0, 1.0)
    return sum(shift)
end

const dosed_panels = @brm begin
    root ~ 0 + (1 | p | site)
    for entry in records
        readings[entry.id] ~ dosed(entry)
    end
    primary ~ Normal(root, 1)
    y ~ Normal(readings, 1)
end

const dosed_explicit = @brm begin
    root ~ 0 + (1 | p | site)
    a ~ 1 + a_dose
    effect(a, Intercept) ~ Normal(0.0, 1.0)
    b ~ 1 + b_dose
    effect(b, Intercept) ~ Normal(0.0, 1.0)
    primary ~ Normal(root, 1)
    ya ~ Normal(sum(a), 1)
    yb ~ Normal(sum(b), 1)
end

# An intercept-only member: RK declares its coefficient as one scalar.
@brm function offset_reading(settings)
    shift ~ 1 + offset(settings.base)
    effect(shift, Intercept) ~ Normal(0.0, 1.0)
    return sum(shift)
end

const offset_panels = @brm begin
    root ~ 0 + (1 | p | site)
    for entry in records
        readings[entry.id] ~ offset_reading(entry)
    end
    primary ~ Normal(root, 1)
    y ~ Normal(readings, 1)
end

const site = ["x", "y"]
const primary = [0.1, -0.1]

# RK and Stan agree on the log density and its gradient through the transport.
function check_transport(m, stan, names)
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    rk, problem = consumer_problem(m)
    transport = brm_coordinate_transport(rk, sb, names)
    n = length(names)
    for u in (zeros(n), collect(range(-0.32, 0.41; length=n)), [0.3sin(i) for i in 1:n])
        value, grad = BridgeStan.log_density_gradient(stan, u; propto=false)
        x = brm_stan_point_to_rk(transport, u)
        rk_value, rk_grad = LogDensityProblems.logdensity_and_gradient(problem, x)
        @test rk_value ≈ brm_stan_logdensity_to_rk(transport, u, value)
        @test rk_grad ≈ brm_stan_gradient_to_rk(transport, u, grad)
    end
    transport
end

@stestset "A member's population coefficients are addressed by its key" begin
    records = [(id=:b, dose=[0.2, 0.9, 0.4]), (id=:a, dose=[0.7, 0.1])]
    input = (; records, site, primary, y=[0.4, 0.2])
    m = dosed_panels(input)
    twin = dosed_explicit((; site, primary, a_dose=records[2].dose,
        b_dose=records[1].dose, ya=[input.y[2]], yb=[input.y[1]]))
    stan = consumer_stan(m, "member-population"; mod=@__MODULE__).model
    ref = consumer_stan(twin, "member-population-twin"; mod=@__MODULE__).model
    names = BridgeStan.param_unc_names(stan)
    twin_names = BridgeStan.param_unc_names(ref)
    @test length(names) == length(twin_names)
    d = brm_descriptor(SBBRMI(m; mod=@__MODULE__, total_groups=()))
    dt = brm_descriptor(SBBRMI(twin; mod=@__MODULE__, total_groups=()))
    # Each member coefficient's coordinate holds the twin's coefficient of the
    # same meaning; the parent block keeps its names.
    perm = zeros(Int, length(names))
    for (key, twin_lp) in ((:a, :a), (:b, :b)),
            (coefficient, twin_coefficient) in ((:Intercept, :Intercept),
                                                (:settings_dose, Symbol(twin_lp, :_dose)))
        r = brm_population_effect_coordinates(d, :shift, names; coefficient,
            submodel=(:readings, key))
        @test r.submodel == (:readings, key)
        @test r.recovered === false && r.link === identity
        t = brm_population_effect_coordinates(dt, twin_lp, twin_names;
            coefficient=twin_coefficient)
        perm[only(r.coordinates)] = only(t.coordinates)
    end
    for (i, name) in enumerate(names)
        startswith(name, "b_p_site") && (perm[i] = findfirst(==(name), twin_names))
    end
    @test sort(perm) == eachindex(names)
    for v in (collect(range(-0.32, 0.41; length=length(names))), [0.3sin(i) for i in eachindex(names)])
        value, grad = BridgeStan.log_density_gradient(stan, v[perm]; propto=false)
        ref_value, ref_grad = BridgeStan.log_density_gradient(ref, v; propto=false)
        @test value ≈ ref_value
        @test grad ≈ ref_grad[perm]
    end
    # Member `a` is the first key in key order: its intercept is the first
    # member's element of the stacked carrier.
    a = brm_population_effect_coordinates(d, :shift, names; submodel=(:readings, :a))
    @test names[only(a.coordinates)] ==
        "readings_result_all_readings_member_all_pop_shift_beta_pop.1.1"
    @test_throws "available labels are" brm_population_effect_coordinates(
        d, :shift, names; coefficient=:dose, submodel=(:readings, :a))
    @test_throws "has no consumer submodel call" brm_population_effect_coordinates(
        d, :shift, names; submodel=(:readings, :c))
    # The member's covariate coefficient vector, indexed by member first on
    # RK, pairs with the twin's meaning, not by matching index spelling.
    transport = check_transport(m, stan, names)
    @test count(p -> p.address.kind === :population, transport.pairs) == 4
end

@stestset "An intercept-only member's coefficient pairs as a scalar" begin
    records = [(id=:b, base=[0.2, 0.9, 0.4]), (id=:a, base=[0.7, 0.1])]
    m = offset_panels((; records, site, primary, y=[0.4, 0.2]))
    stan = consumer_stan(m, "member-population-offset"; mod=@__MODULE__).model
    names = BridgeStan.param_unc_names(stan)
    d = brm_descriptor(SBBRMI(m; mod=@__MODULE__, total_groups=()))
    coordinates = [only(brm_population_effect_coordinates(d, :shift, names;
        submodel=(:readings, key)).coordinates) for key in (:a, :b)]
    @test names[coordinates] == ["readings_result_all_readings_member_all_pop_shift_beta_pop.1.$i"
                                 for i in 1:2]
    transport = check_transport(m, stan, names)
    @test sort([p.address.scope for p in transport.pairs if p.address.kind === :population]) ==
        [(:readings, :a), (:readings, :b)]
end
end
