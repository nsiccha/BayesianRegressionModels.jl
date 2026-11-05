# RUN: julia --project=test test/component_start_values.jl
# RUN: julia --project=test test/component_start_values.jl --stan
# The default run constructs only RK models; it never instantiates Stan.
using Test, BayesianRegressionModels, Distributions, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels

@brm function start_field(x, center, spread)
    a ~ Normal(center, .2)
    sd ~ Exponential(spread)
    bounded ~ Uniform(.1, .9)
    z[1:length(x)] .~ Normal.(0, 1)
    derived = 2 * a
    return x + derived + sd + bounded + z
end

@brm function start_nested(x, center, spread)
    inner ~ start_field(x, center, spread)
    shift ~ Normal(center, .3)
    return inner + shift
end

@brm function start_rate(scale)
    a ~ Exponential(scale)
    return a
end

@brm function start_shapes(x)
    simplex ~ Dirichlet([1., 2., 3.])
    L ~ LKJCholesky(3, 2.)
    return x + simplex[1] + L[2, 1]
end

@brm function start_flat(x)
    a ~ Flat()
    return x + a
end

const START_DATA = (; xl=[-.4, .2, .7], xr=[.3, -.2],
    yl=[.1, .4, .8], yr=[.3, -.1])
const START_BUILDER = @brm begin
    left ~ start_field(xl, center, .5)
    right ~ start_field(xr, -center, 1.5)
    first ~ start_nested(xl, center, .7)
    second ~ start_nested(xr, center, 1.2)
    shape ~ start_shapes(xl)
    yl ~ Normal(left + first + shape, sigma)
    yr ~ Normal(right + second, sigma)
    center ~ Normal(.3, .4)
    sigma ~ Exponential(.6)
end
const START_COMPONENTS = [
    :left => (; a=.7, sd=.4, bounded=.25, z=[-.2, .4, .1]),
    :right => (; a=-.3, z=[.8, -.6]),
    (:first, :inner) => (; a=.9, sd=.2),
    :second => (; shift=-.7),
    :shape => (; simplex=[.2, .3, .5], L=[1. 0 0; .3 sqrt(.91) 0; -.2 .4 sqrt(.8)])]
const START_TOP = (; sigma=.6, center=.8)

function check_component_values(v)
    @test v.sigma ≈ .6
    @test v.center ≈ .8
    for (path, starts) in START_COMPONENTS
        local_values = foldl(getproperty, path isa Symbol ? (path,) : path; init=v)
        for (name, value) in pairs(starts)
            @test getproperty(local_values, name) ≈ value
        end
    end
end

@testset "native component starts and scalar prior medians" begin
    saved = deepcopy((START_DATA, START_TOP, START_COMPONENTS))
    rk = RKBRMI(START_BUILDER(START_DATA))
    base = collect(range(-.3, .4; length=rk.model.layout.total))
    base_saved = copy(base)
    before = BRM._rk_constrained_values(rk, base)
    u = brm_start(rk, START_TOP; component_inits=START_COMPONENTS, base)
    v = BRM._rk_constrained_values(rk, u)
    check_component_values(v)
    @test v.right.sd ≈ before.right.sd
    @test v.right.bounded ≈ before.right.bounded
    @test v.first.inner.z ≈ before.first.inner.z
    for name in keys(before.second.inner)
        @test getproperty(v.second.inner, name) ≈ getproperty(before.second.inner, name)
    end
    @test base == base_saved
    @test isequal((START_DATA, START_TOP, START_COMPONENTS), saved)
    @test brm_start(rk, START_TOP; component_inits=reverse(START_COMPONENTS), base) ≈ u
    @test brm_start(rk, START_TOP; component_inits=Dict(START_COMPONENTS), base) ≈ u

    prior = brm_start(rk, START_TOP; component_inits=START_COMPONENTS, base=:prior_median)
    pv = BRM._rk_constrained_values(rk, prior)
    check_component_values(pv)
    @test pv.right.sd ≈ median(Exponential(1.5))
    @test pv.right.bounded ≈ .5
    @test pv.first.shift ≈ .8
    @test pv.first.inner.bounded ≈ .5
    @test pv.second.inner.a ≈ .8
    @test pv.second.inner.sd ≈ median(Exponential(1.2))
    @test pv.first.inner.z == zeros(3)

    plain = BRM._rk_constrained_values(rk, brm_start(rk, (;); base=:prior_median))
    @test plain.left.a ≈ .3
    @test plain.right.a ≈ -.3
    @test plain.second.inner.a ≈ .3
    @test plain.left.sd ≈ median(Exponential(.5))
    @test plain.right.sd ≈ median(Exponential(1.5))

    for bad in ([:missing => (; a=.1)], [:inner => (; a=.1)],
                [(:first, :missing) => (; a=.1)],
                [:left => (; a=.1), (:left,) => (; a=.2)],
                [:left => (; missing=.1)], [:left => (; x=[1., 2., 3.])],
                [:left => (; derived=.1)],
                [:left => (; a=Inf)], [:left => (; sd=-.1)],
                [:left => (; bounded=1.2)], [:shape => (; simplex=[.2, .3, .6])],
                [:shape => (; L=ones(3, 3))], [:left => (; z=[1., 2.])],
                [:left => (; a=[.1])], [:left => (; z=ones(1, 3))],
                [:left => (; a="bad")], [:left => 1.], ["left" => (; a=.1)])
        @test_throws Exception brm_start(rk, (;); component_inits=bad)
    end

    # Regression for the reported flat/scoped mismatch, with an additional
    # enclosing prior that depends on the component's returned scalar.
    builder = @brm begin
        flat ~ Exponential(.5)
        scoped ~ start_rate(.5)
        dependent ~ Exponential(scoped)
        y ~ Normal(flat + scoped + dependent, .2)
    end
    probe = RKBRMI(builder((; y=[.1, .2])))
    zero = BRM._rk_constrained_values(probe, brm_start(probe, (;)))
    @test zero.flat == zero.scoped.a == 1.
    median_values = BRM._rk_constrained_values(probe, brm_start(probe, (;); base=:prior_median))
    @test median_values.flat ≈ median_values.scoped.a ≈ median(Exponential(.5))
    @test median_values.dependent ≈ median(Exponential(median(Exponential(.5))))
    explicit = BRM._rk_constrained_values(probe,
        brm_start(probe, (;); component_inits=[:scoped => (; a=.7)], base=:prior_median))
    @test explicit.dependent ≈ median(Exponential(.7))

    unsupported = @brm begin
        field ~ start_flat(xl)
        yl ~ Normal(field, 1.)
    end
    flat = RKBRMI(unsupported(START_DATA))
    @test_throws "explicit initializer" brm_start(flat, (;); base=:prior_median)
    fv = BRM._rk_constrained_values(flat, brm_start(flat, (;);
        component_inits=[:field => (; a=.4)], base=:prior_median))
    @test fv.field.a ≈ .4
end

if "--stan" in ARGS
    using BridgeStan
    @testset "component starts: independent compiled Stan readback" begin
        model = START_BUILDER(START_DATA)
        rk = RKBRMI(model)
        sb = SBBRMI(model; mod=@__MODULE__, total_groups=())
        stan = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "component-start.stan")).model
        d = brm_descriptor(sb)
        names = String.(BridgeStan.param_names(stan))
        code = BRM.stan_code(sb)
        function readback(u, path, name)
            constrained = BridgeStan.param_constrain(stan, u)
            coordinates = brm_output_coordinates(d, name, names; submodel=path)
            constrained[coordinates]
        end
        transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan))
        for base in (zeros(rk.model.layout.total), :prior_median)
            r = brm_start(rk, START_TOP; component_inits=START_COMPONENTS, base)
            s = brm_start(sb, stan, START_TOP; component_inits=START_COMPONENTS, base)
            @test brm_rk_point_to_stan(transport, r) ≈ s atol=1e-10 rtol=1e-10
            for (path, starts) in START_COMPONENTS, (name, value) in pairs(starts)
                @test readback(s, path, name) ≈ (value isa Real ? [value] : vec(value))
            end
            @test isfinite(BridgeStan.log_density(stan, s))
            if base === :prior_median
                @test only(readback(s, :right, :sd)) ≈ median(Exponential(1.5))
                @test only(readback(s, (:second, :inner), :a)) ≈ .8
                @test only(readback(s, (:second, :inner), :sd)) ≈ median(Exponential(1.2))
            end
        end
        @test BRM.stan_code(sb) == code
        for bad in ([:missing => (; a=.1)], [:inner => (; a=.1)],
                [:left => (; missing=.1)], [:left => (; x=[1., 2., 3.])],
                [:left => (; derived=.1)],
                [:left => (; sd=-.1)], [:left => (; bounded=1.2)],
                [:shape => (; simplex=[.2, .3, .6])], [:shape => (; L=ones(3, 3))],
                [:left => (; z=[1., 2.])], [:left => (; a=[.1])],
                [:left => (; z=ones(1, 3))])
            @test_throws Exception brm_start(sb, stan, (;); component_inits=bad)
        end
    end
end
