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

# The shape of a joint endpoint block: an intercept beside the margin the member
# puts in the parent's block, and a second predictor with its own covariate.
# Each member centres its intercept prior on its own value.
@brm function marginal(settings)
    base ~ 1 + (1 | parentblock(:p) | settings.site)
    effect(base, Intercept) ~ Normal(settings.center, 0.7)
    slope ~ 1 + settings.dose
    effect(slope, Intercept) ~ Normal(0.0, 1.0)
    return sum(base) + sum(slope)
end

const marginal_panels = @brm begin
    root ~ 0 + (1 | p | site)
    for entry in records
        readings[entry.id] ~ marginal(entry)
    end
    primary ~ Normal(root, 1)
    y ~ Normal(readings, 1)
end

const marginal_explicit = @brm begin
    root ~ 0 + (1 | p | site)
    a ~ 1 + (1 | p | a_site)
    effect(a, Intercept) ~ Normal(0.3, 0.7)
    a_slope ~ 1 + a_dose
    effect(a_slope, Intercept) ~ Normal(0.0, 1.0)
    b ~ 1 + (1 | p | b_site)
    effect(b, Intercept) ~ Normal(-0.4, 0.7)
    b_slope ~ 1 + b_dose
    effect(b_slope, Intercept) ~ Normal(0.0, 1.0)
    primary ~ Normal(root, 1)
    ya ~ Normal(sum(a) + sum(a_slope), 1)
    yb ~ Normal(sum(b) + sum(b_slope), 1)
end

const site = ["x", "y"]
const primary = [0.1, -0.1]

# The twin coordinate of each collection coordinate: a member coefficient by
# its keyed address, given as `(logical, coefficient) => key -> (twin logical,
# twin coefficient)`, and the shared parent block by its name. The two models
# then agree on the log density and its gradient at the permuted point.
function check_member_twin(m, twin, name, coefficients)
    stan = consumer_stan(m, "member-population-" * name; mod=@__MODULE__).model
    ref = consumer_stan(twin, "member-population-twin-" * name; mod=@__MODULE__).model
    names = BridgeStan.param_unc_names(stan)
    twin_names = BridgeStan.param_unc_names(ref)
    @test length(names) == length(twin_names)
    d = brm_descriptor(SBBRMI(m; mod=@__MODULE__, total_groups=()))
    dt = brm_descriptor(SBBRMI(twin; mod=@__MODULE__, total_groups=()))
    perm = zeros(Int, length(names))
    for key in (:a, :b), ((logical, coefficient), twin_address) in coefficients
        r = brm_population_effect_coordinates(d, logical, names; coefficient,
            submodel=(:readings, key))
        @test r.submodel == (:readings, key)
        @test r.recovered === false && r.link === identity
        twin_logical, twin_coefficient = twin_address(key)
        t = brm_population_effect_coordinates(dt, twin_logical, twin_names;
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
    (; stan, names, d)
end

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
    # Each member coefficient's coordinate holds the twin's coefficient of the
    # same meaning; the parent block keeps its names.
    (; stan, names, d) = check_member_twin(m, twin, "dosed", (
        (:shift, :Intercept) => key -> (key, :Intercept),
        (:shift, :settings_dose) => key -> (key, Symbol(key, :_dose))))
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

@stestset "A member's intercept beside its parent-block margin is addressed by its key" begin
    records = [(id=:b, center=-0.4, site=["y", "y", "x"], dose=[0.2, 0.9, 0.4]),
               (id=:a, center=0.3, site=["x", "y"], dose=[0.7, 0.1])]
    input = (; records, site, primary, y=[0.4, 0.2])
    m = marginal_panels(input)
    twin = marginal_explicit((; site, primary, a_site=records[2].site,
        a_dose=records[2].dose, b_site=records[1].site, b_dose=records[1].dose,
        ya=[input.y[2]], yb=[input.y[1]]))
    (; stan, names, d) = check_member_twin(m, twin, "marginal", (
        (:base, :Intercept) => key -> (key, :Intercept),
        (:slope, :Intercept) => key -> (Symbol(key, :_slope), :Intercept),
        (:slope, :settings_dose) => key -> (Symbol(key, :_slope), Symbol(key, :_dose))))
    # A consumer reading draws passes constrained names. The parent block's
    # Cholesky factor has more constrained than unconstrained entries, so the
    # coefficient sits at another position there.
    cnames = BridgeStan.param_names(stan)
    @test length(cnames) > length(names)
    for key in (:a, :b)
        u = brm_population_effect_coordinates(d, :base, names; submodel=(:readings, key))
        c = brm_population_effect_coordinates(d, :base, cnames; submodel=(:readings, key))
        @test cnames[only(c.coordinates)] == names[only(u.coordinates)]
    end
    # Each member's intercept prior is centred on its own value, so a swap
    # of the members' intercepts would change the density above.
    transport = check_transport(m, stan, names)
    @test sort([(p.address.scope, p.address.predictor, p.address.coefficient)
                for p in transport.pairs if p.address.kind === :population]) ==
        sort([((:readings, key), predictor, coefficient) for key in (:a, :b)
              for (predictor, coefficient) in ((:base, :Intercept), (:slope, :Intercept),
                                               (:slope, :settings_dose))])
end

# A calibration survey can select a different design for the level and gain
# of each instrument, while sharing site effects across all instruments.
@brm function selected_calibration(entry)
    level ~ rhs(entry.level_formula) + (1 | parentblock(:p) | entry.site)
    gain ~ rhs(entry.gain_formula) + offset(entry.offset)
    effect(level, :) ~ Normal(0, entry.width)
    effect(level, Intercept) ~ Normal(0, 1)
    optional(effect(gain, :entry_power)) ~ Normal(0, 0.3)
    effect(gain, Intercept) ~ Normal(0, 1)
    if entry.curve
        drift ~ 0 + hsgp(entry.time; k=3, c=1.5)
        drift_sum = sum(drift)
    else
        drift_sum = 0.0
    end
    return sum(level) + sum(gain) + drift_sum
end

@brm function explicit_calibration(entry)
    if entry.curve
        level ~ 1 + standardize(entry.temp) + (1 | parentblock(:p) | entry.site)
        effect(level, :standardize_entry_temp) ~ Normal(0, entry.width)
    else
        level ~ 1 + (1 | parentblock(:p) | entry.site)
    end
    effect(level, Intercept) ~ Normal(0, 1)
    if entry.no_power
        gain ~ 1 + offset(entry.offset)
    else
        gain ~ 1 + entry.power + offset(entry.offset)
        effect(gain, :entry_power) ~ Normal(0, 0.3)
    end
    effect(gain, Intercept) ~ Normal(0, 1)
    if entry.curve
        drift ~ 0 + hsgp(entry.time; k=3, c=1.5)
        drift_sum = sum(drift)
    else
        drift_sum = 0.0
    end
    return sum(level) + sum(gain) + drift_sum
end

function calibration_survey(component)
    brm(quote
        root ~ 0 + (1 | p | site)
        for entry in records
            readings[entry.id] ~ $component(entry)
        end
        primary ~ Normal(root, 1)
        y ~ Normal(readings, 1)
    end; mod=@__MODULE__)
end

@stestset "Record formulas select each predictor and freeze its member's transform" begin
    a = (; id=:a, level_formula="1 + standardize(temp)", gain_formula=1,
        site=["x","y","x"], temp=[1.,2.,3.], width=0.2, offset=zeros(3),
        curve=true, time=[0.,1.,4.], no_power=true)
    b = (; a..., id=:b, level_formula=:(1 + standardize(temp)), temp=[8.,10.,12.], width=0.6)
    c = (; id=:c, level_formula=1, gain_formula="1 + power", site=["x","y"],
        offset=zeros(2), power=[0.2, 0.9], width=0.5, curve=false, no_power=false)
    d = (; id=:d, level_formula=1, gain_formula=1, site=["x","y"], offset=zeros(2),
        width=0.5, curve=false, no_power=true)
    frame(records) = (; records, site, primary, y=[0.3, 0.1, -0.2, 0.4])
    builder = calibration_survey(selected_calibration)
    data = frame((d, b, c, a))
    m = builder(data)
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    desc = brm_descriptor(sb)
    rk = RKBRMI(m)
    # The two selected members share a program and differ in their actual fits.
    for fit in (sb, generative_plan(sb), desc, rk)
        for (key, center, scale) in ((:a, 2., 1.), (:b, 10., 2.))
            submodel = (:readings, key)
            t = only(brm_transform_constants(fit; submodel))
            @test t == brm_transform_constants(fit, :standardize_entry_temp; submodel)
            @test (t.column, t.center, t.scale) == (:entry_temp, center, scale)
        end
        @test isempty(brm_transform_constants(fit; submodel=(:readings, :d)))
        @test_throws "Fitted transforms: ()" brm_transform_constants(fit,
            :standardize_entry_temp; submodel=(:readings, :c))
        @test_throws "unknown record member" brm_transform_constants(fit; submodel=(:readings, :other))
    end
    next = frame((c, merge(a, (; temp=[30.,40.,50.])), d, merge(b, (; temp=[1.,5.,9.]))))
    for fit in (sb, rk)
        for (freeze, ca, sa, cb, sb_) in ((true, 2., 1., 10., 2.), (false, 40., 10., 5., 4.))
            replay = reprocess(fit, next; freeze_constants=freeze)
            for (key, center, scale) in ((:a, ca, sa), (:b, cb, sb_))
                actual = only(brm_transform_constants(replay; submodel=(:readings, key)))
                @test (actual.center, actual.scale) == (center, scale)
            end
        end
        # The member count and coefficient count stay fixed, but a different
        # covariate gives that coefficient a different meaning.
        changed = frame((d, b, merge(c, (; gain_formula="1 + load", load=c.power)), a))
        @test_throws "build a new model" reprocess(fit, changed)
    end
    for bad in ("(", "1; 2", :(run(temp)), :(x ~ Normal(0,1)), true)
        @test_throws ArgumentError builder(frame((d, b, c, merge(a, (; level_formula=bad)))))
    end

    twin = calibration_survey(explicit_calibration)(data)
    stan = consumer_stan(m, "selected-calibration"; mod=@__MODULE__).model
    ref = consumer_stan(twin, "explicit-calibration"; mod=@__MODULE__).model
    names = BridgeStan.param_unc_names(stan)
    # Compare each backend to the written-out component. Identify each native
    # program by its keys; its backend packs its members in stable key order.
    function keyed_names(model, names)
        replacements = Pair{String,String}[]
        for axis in BRM._brm_collection_axes(model)
            result, member = string.(axis.path)
            keys = join(axis.keys, "_")
            append!(replacements, (result => "result_" * keys, member => "member_" * keys))
        end
        [replace(string(n), replacements...) for n in names]
    end
    function check_twin(selected_names, reference_names, selected_gradient, reference_gradient)
        labels = keyed_names(m, selected_names)
        twin_labels = keyed_names(twin, reference_names)
        @test issetequal(labels, twin_labels) && allunique(labels)
        perm = [only(findall(==(n), twin_labels)) for n in labels]
        for u in (collect(range(-0.25, 0.28; length=length(labels))), [0.16cos(i) for i in eachindex(labels)])
            actual, grad = selected_gradient(u[perm])
            expected, ref_grad = reference_gradient(u)
            @test actual ≈ expected
            @test grad ≈ ref_grad[perm]
        end
    end
    check_twin(names, BridgeStan.param_unc_names(ref),
        u -> BridgeStan.log_density_gradient(stan, u; propto=false),
        u -> BridgeStan.log_density_gradient(ref, u; propto=false))
    ref_rk = RKBRMI(twin)
    problem(fit) = rk_logdensity_problem(fit; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    p, ref_p = problem(rk), problem(ref_rk)
    check_twin(coordinate_names(rk.model.layout), coordinate_names(ref_rk.model.layout),
        u -> LogDensityProblems.logdensity_and_gradient(p, u),
        u -> LogDensityProblems.logdensity_and_gradient(ref_p, u))

    # The full models above include each instrument's smooth. Compare the two
    # backends directly for all selected covariates and shared site effects
    # on a survey without smooths, whose complete coordinate transport exists.
    plain = builder(frame(map(r -> (; r..., curve=false), data.records)))
    plain_stan = consumer_stan(plain, "selected-calibration-no-smooth"; mod=@__MODULE__).model
    transport = check_transport(plain, plain_stan, BridgeStan.param_unc_names(plain_stan))
    expected = [(r.id, target, :Intercept) for r in data.records for target in (:level, :gain)]
    append!(expected, [(:a, :level, :standardize_entry_temp),
                       (:b, :level, :standardize_entry_temp), (:c, :gain, :entry_power)])
    @test Set((p.address.scope[2], p.address.predictor, p.address.coefficient)
        for p in transport.pairs if p.address.kind === :population) == Set(expected)
    for (key, target, coefficient) in expected
        r = brm_population_effect_coordinates(desc, target, names; coefficient, submodel=(:readings, key))
        @test length(r.coordinates) == 1
    end
end
end
