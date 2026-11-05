# The Laplace default start of a native gmrf field: `brm_laplace_problem`,
# `brm_laplace_plan`'s diagnostic and `brm_laplace_mode` read the same prior
# medians as `brm_start(rk, (;); base=:prior_median)`, for declared laws, a
# known hyperparameter and a law reading a model-scope parameter.
# RUN: julia --project=test test/gmrf_laplace_start.jl
include("gmrf_rk_cell_models.jl")
using Optim
import LogDensityProblems as LDP

const START_AD = DI.AutoEnzyme(; mode=Enzyme.Reverse)
const FIELD = :gmrf_field_east_north_time

# The medians on the problem's population coordinates, read from the full start.
function full_start_medians(rk, target)
    names = BRM._rk_layout_coordinate_names(rk)
    position = Dict(n => i for (i, n) in enumerate(names))
    brm_start(rk, (;); base=:prior_median)[[position[n] for n in target.names]]
end

@testset "Laplace default start of a model-scope gmrf field" begin
    input = merge(data, (; y=[0.4, -0.3, 0.2, 0.1]))
    declared = @brm input begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(0.5, 0.8)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(0.7, 0.6)
        y ~ Normal(field, 0.2)
    end
    known = @brm input begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        length_scale(field, gmrf((east, north), time), space) = 0.4
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(0.7, 0.6)
        y ~ Normal(field, 0.2)
    end
    reader = @brm input begin
        s0 ~ LogNormal(1, 0.5)
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(s0)
        y ~ Normal(field, 0.2)
    end
    scaled_input = merge(input, (; rate=0.7))
    scaled = @brm scaled_input begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(rate)
        y ~ Normal(field, 0.2)
    end
    expected = (
        ([:sigma, :spatial_length, :temporal_length], declared,
            log.([median(Exponential(0.7)), exp(0.5), exp(0.7)])),
        ([:sigma, :spatial_length, :temporal_length], scaled,
            [log(median(Exponential(0.7))), 0.0, 0.0]),
        ([:sigma, :temporal_length], known, [0.0, 0.7]),
        ([:sigma, :spatial_length, :temporal_length, :s0], reader,
            [log(median(Exponential(exp(1)))), 0.0, 0.0, 1.0]))
    for (locals, model, values) in expected
        rk, _, _ = gmrf_rk_build(model)
        target = brm_laplace_problem(rk; ad_backend=START_AD, integrate=(FIELD,))
        medians = BRM._brm_laplace_prior_median(target)
        names = [l === :s0 ? :s0 : Symbol(FIELD, ".", l) for l in locals]
        @test target.names == names
        @test medians ≈ values
        @test medians ≈ full_start_medians(rk, target)
        @test isfinite(LDP.logdensity(target, medians))
    end
    # A law's inner call reaches the start in the DSL's broadcast spelling.
    spelled = @brm input begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.2), 0.5)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(2 * log(5.0) - log(5.0), 0.5)
        y ~ Normal(field, 0.2)
    end
    rk, _, _ = gmrf_rk_build(spelled)
    target = brm_laplace_problem(rk; ad_backend=START_AD, integrate=(FIELD,))
    @test BRM._brm_laplace_prior_median(target) ≈ log.([median(Exponential(0.7)), 0.2, 5.0])
    # The default mode search starts from these medians: an explicit init of
    # the same point takes the identical path.
    rk, _, _ = gmrf_rk_build(declared)
    target = brm_laplace_problem(rk; ad_backend=START_AD, integrate=(FIELD,))
    medians = BRM._brm_laplace_prior_median(target)
    @test brm_laplace_mode(target).draws == brm_laplace_mode(target; init=medians).draws
end

@testset "brm_laplace_plan measures a count gmrf field at its medians" begin
    input = merge(data, (; count=[1, 0, 2, 1]))
    # The declared log link establishes the affine log-rate structurally;
    # an arbitrary exp(field) expression deliberately stays sampled.
    model = @brm input begin
        log(field) ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(0.2)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(0.5, 0.8)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(0.7, 0.6)
        count ~ Poisson(field)
    end
    rk, _, _ = gmrf_rk_build(model)
    target = brm_laplace_problem(rk; ad_backend=START_AD, integrate=(FIELD,))
    medians = BRM._brm_laplace_prior_median(target)
    @test medians ≈ log.([median(Exponential(0.2)), exp(0.5), exp(0.7)])
    ess(; kwargs...) = only(brm_laplace_plan(rk; ad_backend=START_AD, draws=64, kwargs...).terms).importance_ess
    default, at_medians, at_unit = ess(), ess(; at=medians), ess(; at=zeros(3))
    @test default !== nothing
    @test default.min == at_medians.min && default.median == at_medians.median
    # Positive control: the diagnostic does move with its population point.
    @test default.median != at_unit.median
    @info "gmrf count field importance ESS" default at_unit
end
