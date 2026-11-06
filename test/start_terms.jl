# RUN: julia --project=test test/start_terms.jl
using Test, BayesianRegressionModels, BridgeStan, Distributions, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels
include("testset_filter.jl")
const CACHE = mktempdir()

function start_fixture(builder, data, name)
    brmi = builder(data)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    model = BRM.stan_instantiate(sb; path=joinpath(CACHE, "$name.stan")).model
    d = brm_descriptor(sb)
    names = String.(BridgeStan.param_names(model))
    (; rk, sb, model, d, names)
end

function term_values(f, u, lp, term, parameter)
    c = BridgeStan.param_constrain(f.model, u)
    c[brm_term_coordinates(f.d, lp, f.names; term, parameter).coordinates]
end

function start_pair(f, inits=(;); kwargs...)
    r = brm_start(f.rk, inits; kwargs...)
    s = brm_start(f.sb, f.model, inits; kwargs...)
    t = brm_coordinate_transport(f.rk, f.sb, BridgeStan.param_unc_names(f.model))
    @test brm_rk_point_to_stan(t, r) ≈ s atol=1e-10 rtol=1e-10
    r, s
end

@stestset "term starts: Atlas scale and prior base" begin
    builder = @brm begin
        mu ~ 1 + cdar(year; by=kreis, cor=I_kreis)
        effect(mu, Intercept) ~ Normal(0, 0.02)
        sd(mu, cdar(year)) ~ LogNormal(-4.5, 0.75)
        ar(mu, cdar(year)) ~ Beta(2, 2)
        y ~ Normal(mu, 0.01)
    end
    data = (; year=repeat([2022., 2023., 2024.], inner=2),
        kreis=repeat(["a", "b"], 3), I_kreis=Matrix(1.0I, 2, 2),
        y=[.01, -.01, .012, -.012, .013, -.013])
    f = start_fixture(builder, data, "atlas")
    code = BRM.stan_code(f.sb)
    inits = (; mu=(; Intercept=0.03))
    terms = [:(sd(mu, cdar(year))) => 0.02, :(ar(mu, cdar(year))) => 0.6]
    saved = deepcopy(terms)
    _, s = start_pair(f, inits; term_inits=terms)
    @test only(term_values(f, s, :mu, :cdar_mu_year, :sd)) ≈ 0.02
    @test only(term_values(f, s, :mu, :cdar_mu_year, :ar)) ≈ 0.6
    _, default = start_pair(f, inits)
    @test only(term_values(f, default, :mu, :cdar_mu_year, :sd)) ≈ 1
    rb, sb = start_pair(f; base=:prior_median)
    @test only(term_values(f, sb, :mu, :cdar_mu_year, :sd)) ≈ exp(-4.5)
    @test only(term_values(f, sb, :mu, :cdar_mu_year, :ar)) ≈ 0.5
    re, se = start_pair(f, inits; base=:prior_median, term_inits=terms)
    @test brm_start(f.rk, inits; base=rb, term_inits=terms) ≈ re
    @test brm_start(f.sb, f.model, inits; base=sb, term_inits=terms) ≈ se
    @test terms == saved
    @test BRM.stan_code(f.sb) == code
    @test isfinite(BridgeStan.log_density(f.model, s))
    for backend in ((args...; kw...) -> brm_start(f.rk, args...; kw...),
                    (args...; kw...) -> brm_start(f.sb, f.model, args...; kw...))
        @test_throws "strictly positive" backend((;); term_inits=[:(sd(mu, cdar(year))) => 0])
        @test_throws "strictly between" backend((;); term_inits=[:(ar(mu, cdar(year))) => 1])
        @test_throws "equally specific" backend((;); term_inits=[terms[1], terms[1]])
        @test_throws "matches no" backend((;); term_inits=[:(sd(mu, cdar(typo))) => .1])
        @test_throws "finite" backend((;); term_inits=[:(sd(mu, cdar(year))) => Inf])
        @test_throws DimensionMismatch backend((;); term_inits=[:(sd(mu, cdar(year))) => [.1, .2]])
        @test_throws "unknown base" backend((;); base=:typo)
    end
end

@stestset "term starts: specificity and qualified repeated terms" begin
    builder = @brm begin
        a ~ 0 + hsgp(x; k=3)
        b ~ 0 + hsgp(x; k=3)
        y ~ Normal(a + b, 0.2)
    end
    data = (; x=collect(range(-1, 1; length=8)), y=zeros(8))
    f = start_fixture(builder, data, "repeated")
    terms = [:(sd(:, hsgp(x))) => .2, :(sd(b, hsgp(x))) => .4]
    _, s = start_pair(f; term_inits=terms)
    @test only(term_values(f, s, :a, :hsgp_x, :sd)) ≈ .2
    @test only(term_values(f, s, :b, :hsgp_x, :sd)) ≈ .4
    _, reversed = start_pair(f; term_inits=reverse(terms))
    @test reversed ≈ s

    builder = @brm begin
        mu ~ 0 + hsgp(x; k=3) + hsgp(x; k=2, cov=:periodic, period=2.0)
        y ~ Normal(mu, 0.2)
    end
    f = start_fixture(builder, data, "qualified")
    @test_throws "ambiguous" brm_start(f.rk, (;); term_inits=[:(sd(mu, hsgp(x))) => .2])
    @test_throws "ambiguous" brm_start(f.sb, f.model, (;); term_inits=[:(sd(mu, hsgp(x))) => .2])
    terms = [
        :(sd(mu, hsgp(x; cov=:exp_quad))) => .2,
        :(sd(mu, hsgp(x; cov=:periodic))) => .4]
    r = brm_start(f.rk, (;); term_inits=terms)
    s = brm_start(f.sb, f.model, (;); term_inits=terms)
    # The existing transport addresses HSGPs by axes, so repeated axes need
    # independent read-back from each emitter's distinct parameters.
    v = BRM._rk_constrained_values(f.rk, r)
    @test [v.hsgp_x.sigma, v.hsgp_x_2.sigma] ≈ [.2, .4]
    c = BridgeStan.param_constrain(f.model, s)
    @test c[[findfirst(==("hsgp_x_sigma"), f.names),
        findfirst(==("hsgp_mu_x_sigma"), f.names)]] ≈ [.2, .4]
    @test isfinite(BridgeStan.log_density(f.model, s))
end

@stestset "prior bases: ordinary laws, dependencies and explicit precedence" begin
    builder = @brm begin
        scale ~ LogNormal(-2, .3)
        sigma ~ truncated(Normal(scale, .2); lower=.01, upper=.8)
        mu ~ 1 + x + (1 | u | g)
        effect(mu, Intercept) ~ Normal(2, .1)
        effect(mu, x) ~ Normal(-.3, .2)
        sd(:, u) ~ Exponential(.3)
        y ~ Normal(mu, sigma)
    end
    data = (; x=collect(range(-1, 1; length=6)), g=repeat(["a", "b"], 3), y=zeros(6))
    f = start_fixture(builder, data, "ordinary")
    _, s = start_pair(f; base=:prior_median)
    c = BridgeStan.param_constrain(f.model, s)
    @test c[only(brm_output_coordinates(f.d, :scale, f.names))] ≈ exp(-2)
    @test c[only(brm_output_coordinates(f.d, :sigma, f.names))] ≈
        median(truncated(Normal(exp(-2), .2), .01, .8))
    @test c[only(brm_ranef_sd_coordinates(f.d, :mu, f.names; id=:u).coordinates)] ≈ median(Exponential(.3))
    _, s = start_pair(f, (; scale=.4, mu=(; Intercept=3.), u=(; sd=.1)); base=:prior_median)
    c = BridgeStan.param_constrain(f.model, s)
    @test c[only(brm_output_coordinates(f.d, :sigma, f.names))] ≈
        median(truncated(Normal(.4, .2), .01, .8))

    builder = @brm begin
        raw_sigma ~ Flat()
        mu ~ 1 + x
        y ~ Normal(mu, exp(raw_sigma))
    end
    f = start_fixture(builder, data, "flat")
    @test_throws "explicit initializer" brm_start(f.rk, (;); base=:prior_median)
    @test_throws "explicit initializer" brm_start(f.sb, f.model, (;); base=:prior_median)
    start_pair(f, (; raw_sigma=-2.); base=:prior_median)
end

@stestset "term starts: HSGP floors and known values" begin
    builder = @brm begin
        mu ~ 1 + hsgp(x; k=3)
        sd(mu, hsgp(x)) = 0.2
        sigma ~ Exponential(0.3)
        y ~ Normal(mu, sigma)
    end
    data = (; x=collect(range(-3, 3; length=8)), y=collect(range(-.1, .1; length=8)))
    f = start_fixture(builder, data, "hsgp")
    floor = brm_length_scale_floor(f.d, :mu; term=:hsgp_x)
    _, s = start_pair(f; base=:prior_median,
        term_inits=[:(sd(mu, hsgp(x))) => .2])
    @test only(term_values(f, s, :mu, :hsgp_x, :length_scale)) ≈
        median(truncated(LogNormal(0, 1), floor, Inf))
    c = BridgeStan.param_constrain(f.model, s)
    @test c[only(brm_output_coordinates(f.d, :sigma, f.names))] ≈ median(Exponential(.3))
    for backend in ((args...; kw...) -> brm_start(f.rk, args...; kw...),
                    (args...; kw...) -> brm_start(f.sb, f.model, args...; kw...))
        @test_throws "known at" backend((;); term_inits=[:(sd(mu, hsgp(x))) => .3])
    end
end

@stestset "prior bases: native gmrf fields" begin
    data = (; east=[0.1, 0.9, 0.7, 0.3], north=[0.2, 0.4, 0.8, 0.5],
        time=[0.0, 0.3, 0.3, 1.7], channel=["b", "a", "b", "a"], y=[0.4, -0.3, 0.2, 0.1],
        xy=[0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5], triangles=[1 2 5; 2 3 5; 3 4 5; 4 1 5])
    term = :gmrf_field_east_north_time
    hyper(f, s, parameter) = only(term_values(f, s, :field, term, parameter))
    sd_address = :(sd(field, gmrf((east, north), time)))
    space = :(length_scale(field, gmrf((east, north), time), space))
    time = :(length_scale(field, gmrf((east, north), time), time))

    # Declared laws: each hyperparameter takes its own median on both backends.
    declared = @brm begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(0.5, 0.8)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(0.7, 0.6)
        y ~ Normal(field, 0.2)
    end
    f = start_fixture(declared, data, "gmrf_declared")
    r, s = start_pair(f; base=:prior_median)
    medians = (; sd=median(Exponential(0.7)), spatial_length=exp(0.5), temporal_length=exp(0.7))
    for (parameter, expected) in pairs(medians)
        @test hyper(f, s, parameter) ≈ expected
    end
    rv = getproperty(BRM._rk_constrained_values(f.rk, r), term)
    @test [rv.sigma, rv.spatial_length, rv.temporal_length] ≈ collect(medians)
    # The field itself keeps its neutral coordinates, so the base differs from
    # the unit-scale start only in the three hyperparameters.
    unit = brm_start(f.rk, (;))
    @test findall(!iszero, r .- unit) == 1:3
    @test isfinite(BridgeStan.log_density(f.model, s))
    # Explicit values win over the prior base, one hyperparameter at a time.
    _, explicit = start_pair(f; base=:prior_median, term_inits=[sd_address => 0.3, space => 2.0])
    @test hyper(f, explicit, :sd) ≈ 0.3
    @test hyper(f, explicit, :spatial_length) ≈ 2.0
    @test hyper(f, explicit, :temporal_length) ≈ medians.temporal_length
    @test_throws "strictly positive" brm_start(f.rk, (;); base=:prior_median, term_inits=[time => 0.0])

    # The default law, a law on a known sibling and a restated known value.
    known = @brm begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        length_scale(field, gmrf((east, north), time), space) = 0.4
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(0.7, 0.6)
        y ~ Normal(field, 0.2)
    end
    f = start_fixture(known, data, "gmrf_known")
    _, s = start_pair(f; base=:prior_median)
    @test hyper(f, s, :sd) ≈ median(LogNormal(0, 1))
    @test hyper(f, s, :temporal_length) ≈ medians.temporal_length
    @test_throws "known value 0.4" brm_term_coordinates(f.d, :field, f.names; term, parameter=:spatial_length)
    r_known = brm_start(f.rk, (;); base=:prior_median)
    @test brm_start(f.rk, (;); base=:prior_median, term_inits=[space => 0.4]) ≈ r_known
    for backend in ((args...; kw...) -> brm_start(f.rk, args...; kw...),
                    (args...; kw...) -> brm_start(f.sb, f.model, args...; kw...))
        @test_throws "known at" backend((;); base=:prior_median, term_inits=[space => 0.5])
    end

    # A law reading a model-scope parameter uses that parameter's own start.
    reader = @brm begin
        s0 ~ LogNormal(1, 0.5)
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(s0)
        y ~ Normal(field, 0.2)
    end
    f = start_fixture(reader, data, "gmrf_reader")
    _, s = start_pair(f; base=:prior_median)
    @test hyper(f, s, :sd) ≈ median(Exponential(exp(1)))
    _, chosen = start_pair(f, (; s0=3.0); base=:prior_median)
    @test hyper(f, chosen, :sd) ≈ median(Exponential(3.0))

    # A law reading a data scalar takes its value, as a stated number does.
    scaled = @brm begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(rate)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(location, 0.8)
        y ~ Normal(field, 0.2)
    end
    f = start_fixture(scaled, merge(data, (; rate=0.7, location=0.5)), "gmrf_data_scalar")
    _, s = start_pair(f; base=:prior_median)
    @test hyper(f, s, :sd) ≈ medians.sd
    @test hyper(f, s, :spatial_length) ≈ medians.spatial_length
    @test isfinite(BridgeStan.log_density(f.model, s))

    # A correlated channel axis: the sampled factor keeps its neutral identity.
    correlated = @brm begin
        Omega ~ LKJCholesky(2, 2.0)
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) ~ Exponential(0.7)
        y ~ Normal(field, 0.2)
    end
    f = start_fixture(correlated, data, "gmrf_channels")
    _, s = start_pair(f; base=:prior_median)
    @test only(term_values(f, s, :field, :gmrf_field_east_north_time_channel, :sd)) ≈ medians.sd
    @test isfinite(BridgeStan.log_density(f.model, s))
end

@stestset "term starts: monotonic simplexes and smoothing components" begin
    builder = @brm begin
        mu ~ 0 + mo1(dose)
        y ~ Normal(mu, .2)
    end
    data = (; dose=repeat(1:4, 2), y=zeros(8))
    f = start_fixture(builder, data, "monotonic")
    terms = [:(simplex(mu, mo1(dose))) => [.2, .3, .5]]
    _, s = start_pair(f; term_inits=terms)
    @test term_values(f, s, :mu, :mo1_dose, :simplex) ≈ [.2, .3, .5]
    @test_throws "summing to one" brm_start(f.rk, (;);
        term_inits=[:(simplex(mu, mo1(dose))) => [.2, .3, .4]])
    # A zero simplex coordinate is backend-specific; each valid neutral base
    # is preserved rather than called a multivariate median.
    @test brm_start(f.rk, (;); base=:prior_median) ≈ brm_start(f.rk, (;)) atol=1e-12
    @test brm_start(f.sb, f.model, (;); base=:prior_median) ≈ brm_start(f.sb, f.model, (;)) atol=1e-12

    builder = @brm begin
        mu ~ 0 + t2(x, z; k=(3, 3))
        y ~ Normal(mu, .2)
    end
    data = (; x=repeat(collect(1.:5.), inner=5), z=repeat(collect(1.:5.), 5), y=zeros(25))
    f = start_fixture(builder, data, "tensor")
    terms = [:(sd(mu, t2(x, z), rr)) => .11,
        :(sd(mu, t2(x, z), rn)) => .22, :(sd(mu, t2(x, z), nr)) => .33]
    r = brm_start(f.rk, (;); term_inits=terms)
    s = brm_start(f.sb, f.model, (;); term_inits=terms)
    # Independent emitter coordinates: the three penalties are separate RK
    # scalars and the Stan vector's (rr,rn,nr) entries.
    v = BRM._rk_constrained_values(f.rk, r).t2_x_z
    @test [v.sd1, v.sd2, v.sd3] ≈ [.11, .22, .33]
    c = BridgeStan.param_constrain(f.model, s)
    @test c[[findfirst(==("t2_mu_x_z_sd_pen.$j"), f.names) for j in 1:3]] ≈ [.11, .22, .33]
    @test_throws "three independent" brm_start(f.rk, (;); term_inits=[:(sd(mu, t2(x, z))) => .1])
    rb = brm_start(f.rk, (;); base=:prior_median)
    @test BRM._rk_constrained_values(f.rk, rb).t2_x_z.sd1 ≈ median(truncated(Normal(), 0, Inf))
end

@stestset "prior bases: partially known scales" begin
    builder = @brm begin
        mu ~ 1 + x + (1 + x | p | g)
        effect(mu, Intercept) ~ Normal(-1, .5)
        sd(mu, p, Intercept) = .2
        sd(mu, p, x) ~ Exponential(.3)
        y ~ Normal(mu, .2)
    end
    data = (; g=repeat(["a", "b", "c"], 4), x=collect(range(-1, 1; length=12)), y=zeros(12))
    f = start_fixture(builder, data, "bounded_known")
    _, s = start_pair(f; base=:prior_median)
    c = BridgeStan.param_constrain(f.model, s)
    i = only(brm_population_effect_coordinates(f.d, :mu, f.names; coefficient=:Intercept).coordinates)
    @test c[i] ≈ -1
    scales = only(o for o in f.d.outputs if o.kind === :parameter && o.component === :scale)
    @test c[BRM._brm_element_coordinates(scales, f.names)] ≈ [median(Exponential(.3))]
end

@stestset "prior bases: bounded Stan coefficients" begin
    # Bounded population priors are currently an SB lowering. This checks
    # their actual constrained median without adding an RK lowering.
    data = (; x=collect(range(-1, 1; length=8)), y=zeros(8))
    brmi = @brm data begin
        mu ~ 1 + x
        effect(mu, Intercept) ~ restricted(Normal(-1, .5), .1, .8)
        y ~ Normal(mu, .2)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    model = BRM.stan_instantiate(sb; path=joinpath(CACHE, "bounded_pop.stan")).model
    d = brm_descriptor(sb)
    names = String.(BridgeStan.param_names(model))
    c = BridgeStan.param_constrain(model, brm_start(sb, model, (;); base=:prior_median))
    i = only(brm_population_effect_coordinates(d, :mu, names; coefficient=:Intercept).coordinates)
    @test c[i] ≈ median(truncated(Normal(-1, .5), .1, .8))
end

@stestset "prior bases: per-margin laws in each stratum" begin
    builder = @brm begin
        mu ~ 1 + x + (1 + x | p | gr(g, by=arm))
        sd(mu, p, Intercept) ~ Exponential(.3)
        sd(mu, p, x) ~ Exponential(.7)
        y ~ Normal(mu, .2)
    end
    data = (; g=repeat(["a", "b", "c", "d"], 3),
        arm=repeat(["p", "p", "q", "q"], 3),
        x=collect(range(-1, 1; length=12)), y=zeros(12))
    f = start_fixture(builder, data, "stratum_priors")
    _, s = start_pair(f; base=:prior_median)
    c = BridgeStan.param_constrain(f.model, s)
    scales = only(o for o in f.d.outputs if o.kind === :parameter && o.component === :scale)
    @test reshape(c[BRM._brm_element_coordinates(scales, f.names)], 2, 2) ≈
        repeat([median(Exponential(.3)), median(Exponential(.7))], 1, 2)
end

@stestset "prior bases: exact-total scale laws" begin
    data = (; g=repeat(["a", "b", "c"], 4), x=collect(range(-1, 1; length=12)), y=zeros(12))
    brmi = @brm data begin
        sigma ~ Exponential(.4)
        mu ~ 1 + x + (1 | p | g)
        sd(mu, p) ~ Exponential(.3)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=:auto)
    model = BRM.stan_instantiate(sb; path=joinpath(CACHE, "total_priors.stan")).model
    d = brm_descriptor(sb)
    names = String.(BridgeStan.param_names(model))
    s = brm_start(sb, model, (;); base=:prior_median)
    c = BridgeStan.param_constrain(model, s)
    block = only(total_effect_blocks(sb))
    output = only(o for o in d.outputs if o.name === block.scales)
    @test c[BRM._brm_element_coordinates(output, names)] ≈ [median(Exponential(.3))]
    @test isfinite(BridgeStan.log_density(model, s))
end

@stestset "prior bases: predicted HSGP hyperparameters" begin
    builder = @brm begin
        mu ~ 1 + hsgp(x; k=3, by=g)
        log(length_scale(hsgp(x))) ~ 1 + (1 | g)
        log(sd(hsgp(x))) ~ 1 + (1 | g)
        length_scale(:, hsgp(x)) ~ Normal(.2, .7)
        sd(:, hsgp(x)) ~ Normal(-.1, .8)
        y ~ Normal(mu, 1)
    end
    data = (; x=repeat(collect(range(-1, 1; length=4)), 2),
        g=repeat(["a", "b"], inner=4), y=zeros(8))
    f = start_fixture(builder, data, "hsgp_hyper_priors")
    _, s = start_pair(f; base=:prior_median)
    @test only(term_values(f, s, :mu, :hsgp_x_by_g, :length_scale_intercept)) ≈ .2
    @test only(term_values(f, s, :mu, :hsgp_x_by_g, :sd_intercept)) ≈ -.1
    for role in (:length_scale_ranef_sd, :sd_ranef_sd)
        @test only(term_values(f, s, :mu, :hsgp_x_by_g, role)) ≈ median(truncated(Normal(), 0, Inf))
    end
    @test_throws "predict" brm_start(f.rk, (;); term_inits=[:(sd(mu, hsgp(x))) => .3])
    @test_throws "predict" brm_start(f.sb, f.model, (;); term_inits=[:(sd(mu, hsgp(x))) => .3])
end


@stestset "term starts: multi-axis exact GP integration" begin
    builder = @brm begin
        mu ~ 0 + gp(x, z; iso=false, jitter=.01)
        sd(mu, gp(x, z)) ~ Exponential(.4)
        length_scale(mu, gp(x, z)) ~ LogNormal(-.2, .5)
        y ~ Normal(mu, .2)
    end
    data = (; x=[-.4, .1, .7], z=[.2, -.3, -.6], y=zeros(3))
    f = start_fixture(builder, data, "multiaxis_gp")
    terms = [:(sd(mu, gp(x, z))) => .2,
        :(length_scale(mu, gp(x, z))) => [.3, .7]]
    r = brm_start(f.rk, (;); term_inits=terms)
    s = brm_start(f.sb, f.model, (;); term_inits=terms)
    term = only(t for p in f.rk.plan.predictors for t in p.terms if t.kind === :gp)
    v = BRM._rk_constrained_values(f.rk, r)[term.options.f]
    @test v.rho ≈ [.3, .7]
    @test v.sigma ≈ .2
    c = BridgeStan.param_constrain(f.model, s)
    rho = [findfirst(==("gp_x_z_rho.$j"), f.names) for j in 1:2]
    sigma = findfirst(==("gp_x_z_sigma"), f.names)
    @test c[rho] ≈ [.3, .7]
    @test c[sigma] ≈ .2
    @test isfinite(BridgeStan.log_density(f.model, s))
    rb = brm_start(f.rk, (;); base=:prior_median)
    vb = BRM._rk_constrained_values(f.rk, rb)[term.options.f]
    cb = BridgeStan.param_constrain(f.model,
        brm_start(f.sb, f.model, (;); base=:prior_median))
    @test vb.rho ≈ fill(exp(-.2), 2)
    @test cb[rho] ≈ vb.rho
    @test cb[sigma] ≈ vb.sigma ≈ median(Exponential(.4))
end

@stestset "term starts: ungrouped anisotropic HSGP length scales" begin
    # One prior address reaches both predictors, and every law reads bound data.
    builder = @brm begin
        a ~ 1 + hsgp(x, z, t; iso=false, k=(2, 2, 2),
            domain=((-3., 3.), (-3., 3.), (-3., 3.)))
        b ~ 1 + hsgp(x, z, t; iso=false, k=(2, 2, 2),
            domain=((-3., 3.), (-3., 3.), (-3., 3.)))
        sd(:, hsgp(x, z, t)) ~ Exponential(sd_mean)
        length_scale(:, hsgp(x, z, t)) ~ LogNormal(length_mu, length_sd)
        y_a ~ Normal(a, .4)
        y_b ~ Normal(b, .4)
    end
    data = (; x=[-.3, .4], z=[.1, -.2], t=[-.2, .3], y_a=[.1, -.2], y_b=[.2, .3],
        sd_mean=1., length_mu=.25, length_sd=.7)
    f = start_fixture(builder, data, "anisotropic_hsgp")
    address = :(length_scale(:, hsgp(x, z, t)))
    function rk_lengths(u)
        v = BRM._rk_constrained_values(f.rk, u)
        map(w -> [w.rho_1, w.rho_2, w.rho_3], (v.hsgp_x_z_t, v.hsgp_x_z_t_2))
    end
    stan_lengths(u) = (term_values(f, u, :a, :hsgp_x_z_t, :length_scale),
        term_values(f, u, :b, :hsgp_x_z_t, :length_scale))

    # A scalar broadcasts over the axes, on both predictors.
    inits = (; a=(; Intercept=.1))
    r, s = start_pair(f, inits; term_inits=[address => 1.2])
    @test all(≈(fill(1.2, 3)), rk_lengths(r))
    @test all(≈(fill(1.2, 3)), stan_lengths(s))
    @test isfinite(BridgeStan.log_density(f.model, s))
    # Only the intercept and the six length scales leave the base.
    @test count(!iszero, r .- brm_start(f.rk, (;))) == 7

    # A vector holds one value per axis, in formula order, and a predictor's
    # own address takes precedence over the wildcard.
    r, s = start_pair(f; term_inits=[address => [.8, 1.2, 2.],
        :(length_scale(b, hsgp(x, z, t))) => [3., .5, 1.1]])
    @test rk_lengths(r)[1] ≈ [.8, 1.2, 2.]
    @test rk_lengths(r)[2] ≈ [3., .5, 1.1]
    @test stan_lengths(s)[1] ≈ [.8, 1.2, 2.]
    @test stan_lengths(s)[2] ≈ [3., .5, 1.1]

    for backend in ((args...; kw...) -> brm_start(f.rk, args...; kw...),
                    (args...; kw...) -> brm_start(f.sb, f.model, args...; kw...))
        @test_throws DimensionMismatch backend((;); term_inits=[address => [.8, 1.2]])
        @test_throws "strictly positive" backend((;); term_inits=[address => [.8, 0., 1.2]])
    end

    # The prior base reaches the same carriers; explicit values still win.
    medians = fill(exp(.25), 3)
    r, s = start_pair(f; base=:prior_median)
    @test all(≈(medians), rk_lengths(r))
    @test all(≈(medians), stan_lengths(s))
    r, s = start_pair(f; base=:prior_median,
        term_inits=[:(length_scale(a, hsgp(x, z, t))) => [.8, 1.2, 2.]])
    @test rk_lengths(r)[1] ≈ [.8, 1.2, 2.]
    @test rk_lengths(r)[2] ≈ medians
    @test stan_lengths(s)[2] ≈ medians

    # Default laws keep one validity floor per axis.
    defaults = @brm begin
        mu ~ 0 + hsgp(x, z; iso=false, k=(3, 4))
        y_a ~ Normal(mu, .4)
    end
    f = start_fixture(defaults, data, "anisotropic_hsgp_defaults")
    address = :(length_scale(mu, hsgp(x, z)))
    r, s = start_pair(f; term_inits=[address => [1.5, 2.5]])
    @test term_values(f, s, :mu, :hsgp_x_z, :length_scale) ≈ [1.5, 2.5]
    r, s = start_pair(f; base=:prior_median)
    medians = term_values(f, s, :mu, :hsgp_x_z, :length_scale)
    v = BRM._rk_constrained_values(f.rk, r).hsgp_x_z
    @test [v.rho_1, v.rho_2] ≈ medians

    # A single axis samples one scale, whatever `iso=` says.
    one_axis = @brm begin
        mu ~ 0 + hsgp(x; iso=false, k=3)
        length_scale(mu, hsgp(x)) ~ LogNormal(0, .7)
        y_a ~ Normal(mu, .4)
    end
    f = start_fixture(one_axis, data, "one_axis_iso_false")
    r, s = start_pair(f; term_inits=[:(length_scale(mu, hsgp(x))) => 1.3])
    @test only(term_values(f, s, :mu, :hsgp_x, :length_scale)) ≈ 1.3
    @test BRM._rk_constrained_values(f.rk, r).hsgp_x.rho_iso ≈ 1.3

    # Known axes hold no sampled carrier: they may only be restated.
    known = @brm begin
        mu ~ 0 + hsgp(x, z; iso=false, k=(2, 2))
        length_scale(mu, hsgp(x, z)) = [.9, 1.4]
        y_a ~ Normal(mu, .4)
    end
    f = start_fixture(known, data, "anisotropic_hsgp_known")
    address = :(length_scale(mu, hsgp(x, z)))
    base, _ = start_pair(f; base=:prior_median)
    r, _ = start_pair(f; base=:prior_median, term_inits=[address => [.9, 1.4]])
    @test r ≈ base
    for backend in ((args...; kw...) -> brm_start(f.rk, args...; kw...),
                    (args...; kw...) -> brm_start(f.sb, f.model, args...; kw...))
        @test_throws "known at" backend((;); term_inits=[address => [.9, 1.5]])
    end
end

@stestset "prior bases: derived CAR and CDAR inputs" begin
    source = quote
        area_sub = area[rows]
        time_sub = time[rows]
        group_sub = group[rows]
        mu ~ 0 + car(area_sub; edges=(from, to), type=:icar) +
            cdar(time_sub; by=group_sub, cor=C)
        sd(mu, car(area_sub)) ~ Normal(0, .4)
        sd(mu, cdar(time_sub)) ~ LogNormal(-3, .2)
        ar(mu, cdar(time_sub)) ~ Beta(2, 2)
        y ~ Normal(mu, 1)
    end
    native = @brm begin
        area_sub = area[rows]
        time_sub = time[rows]
        group_sub = group[rows]
        mu ~ 0 + car(area_sub; edges=(from, to), type=:icar) +
            cdar(time_sub; by=group_sub, cor=C)
        sd(mu, car(area_sub)) ~ Normal(0, .4)
        sd(mu, cdar(time_sub)) ~ LogNormal(-3, .2)
        ar(mu, cdar(time_sub)) ~ Beta(2, 2)
        y ~ Normal(mu, 1)
    end
    data = (; area=[1, 2, 3, 1, 2, 3], time=[1., 1., 1., 2., 2., 2.],
        group=[1, 2, 1, 1, 2, 1], rows=[1, 2, 4, 5],
        from=[1, 2], to=[2, 3], C=Matrix(1.0I, 2, 2), y=zeros(4))
    for (j, builder) in enumerate((native, brm(source; mod=@__MODULE__)))
        f = start_fixture(builder, data, "derived_term_inputs_$j")
        _, s = start_pair(f; base=:prior_median)
        @test only(term_values(f, s, :mu, :car_mu_area_sub, :sd)) ≈
            median(truncated(Normal(0, .4), 0, Inf))
        @test only(term_values(f, s, :mu, :cdar_mu_time_sub, :sd)) ≈ exp(-3)
        @test only(term_values(f, s, :mu, :cdar_mu_time_sub, :ar)) ≈ .5
        @test isfinite(BridgeStan.log_density(f.model, s))
        _, explicit = start_pair(f; base=:prior_median,
            term_inits=[:(sd(mu, car(area_sub))) => .15])
        @test only(term_values(f, explicit, :mu, :car_mu_area_sub, :sd)) ≈ .15

        # Stan replay preserves the fitted graph and time grid. RK requires
        # a fresh rebuild for CDAR row latents; both starts see the new subset.
        fresh = merge(data, (; rows=[2, 4], y=zeros(2)))
        rr = reprocess(f.rk, fresh; freeze_constants=false)
        ss = reprocess(f.sb, fresh)
        model = BRM.stan_instantiate(ss;
            path=joinpath(CACHE, "derived_term_inputs_$j.stan")).model
        replay = (; rk=rr, sb=ss, model, d=brm_descriptor(ss),
            names=String.(BridgeStan.param_names(model)))
        # Replays own distinct BRMI objects, so transport cannot compare them.
        # Read each backend's constrained parameter independently instead.
        rv = BRM._rk_constrained_values(rr, brm_start(rr, (;); base=:prior_median))
        car = only(r for r in BRM._rk_coordinate_records(rr.plan) if r.kind === :car)
        @test getproperty(rv[car.declaration], car.locals.sigma) ≈
            median(truncated(Normal(0, .4), 0, Inf))
        s = brm_start(ss, model, (;); base=:prior_median)
        @test only(term_values(replay, s, :mu, :car_mu_area_sub, :sd)) ≈
            median(truncated(Normal(0, .4), 0, Inf))
        @test isfinite(BridgeStan.log_density(model, s))
    end
end

@stestset "prior bases: fitted constants and deterministic dependencies" begin
    # Atlas's full model constructs this prior through brm(source), whose
    # constant evaluator is an interpreted closure. It is resolved once by
    # the backend and remains a fitted value on a prediction replay.
    source = quote
        m = constant(log(sum(population) / sum(poi)))
        shifted = exp(m) + delta
        log_weight ~ Normal(m, .2)
        scale ~ LogNormal(shifted, .1)
        mu ~ 1
        effect(mu, Intercept) ~ Normal(m, .2)
        y ~ Normal(mu + log_weight, scale)
    end
    native = @brm begin
        m = constant(log(sum(population) / sum(poi)))
        shifted = exp(m) + delta
        log_weight ~ Normal(m, .2)
        scale ~ LogNormal(shifted, .1)
        mu ~ 1
        effect(mu, Intercept) ~ Normal(m, .2)
        y ~ Normal(mu + log_weight, scale)
    end
    data = (; population=[2., 4., 6., 8.], poi=[1., 2., 1., 1.],
        delta=.3, y=zeros(4))
    for (j, builder) in enumerate((native, brm(source; mod=@__MODULE__)))
        f = start_fixture(builder, data, "constant_dependency_$j")
        _, s = start_pair(f; base=:prior_median)
        c = BridgeStan.param_constrain(f.model, s)
        @test c[only(brm_output_coordinates(f.d, :log_weight, f.names))] ≈ log(4)
        @test c[only(brm_output_coordinates(f.d, :scale, f.names))] ≈ exp(4.3)
        @test c[only(brm_population_effect_coordinates(f.d, :mu, f.names;
            coefficient=:Intercept).coordinates)] ≈ log(4)
        @test isfinite(BridgeStan.log_density(f.model, s))

        # Different raw inputs must not replace the fitted constant. A fresh
        # fit, in contrast, takes the new input ratio.
        fresh = (; population=[20., 40.], poi=[1., 1.], delta=.3, y=zeros(2))
        rr = reprocess(f.rk, fresh)
        ur = brm_start(rr, (;); base=:prior_median)
        @test BRM._rk_constrained_values(rr, ur).log_weight ≈ log(4)
        replay = reprocess(f.sb, fresh)
        model = BRM.stan_instantiate(replay;
            path=joinpath(CACHE, "constant_dependency_$j.stan")).model
        cs = BridgeStan.param_constrain(model,
            brm_start(replay, model, (;); base=:prior_median))
        names = String.(BridgeStan.param_names(model))
        @test cs[only(brm_output_coordinates(brm_descriptor(replay), :log_weight, names))] ≈ log(4)
        restored = RKBRMI(builder(fresh), brm_fitted_state(f.rk))
        @test BRM._rk_constrained_values(restored,
            brm_start(restored, (;); base=:prior_median)).log_weight ≈ log(4)
        fresh_rk = RKBRMI(builder(fresh))
        @test BRM._rk_constrained_values(fresh_rk,
            brm_start(fresh_rk, (;); base=:prior_median)).log_weight ≈ log(30)
    end
end
