# Eligibility only: no AD preparation, optimization, or posterior sampling.
# Run independently with --project=test.
using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels
isdefined(BRM, :_brm_svi_gaussian_certificate) ||
    Base.include(BRM, joinpath(@__DIR__, "../src/nlme_svi_laplace_plan.jl"))

function gaussian_certificate_fixture(; observation=:(y[p] ~ Normal(mu[p], 0.7)),
        population=:(effect(mu, Intercept) ~ Normal(0, 1)),
        extra=Any[], body=Any[], linked=false, direct=false)
    data = (; probe=repeat(1:3; inner=3), time=repeat([.1, .8, 1.6], 3),
        sx=[-.7,.2,1.1], sy=[.3,-.8,.6], y=collect(range(-1.,1.;length=9)))
    predictor = linked ? :(log(mu) ~ 1 + (1 | probe) +
        hsgp(time; k=2, by=probe, cor=multigroup((sx,sy), 2.; k=2))) :
        :(mu ~ 1 + (1 | probe) + hsgp(time; k=2, by=probe, cor=multigroup((sx,sy), 2.; k=2)))
    # Construct a real statement list: interpolating `nothing` creates an
    # executable plate statement and tests a different model.
    plate = :(@plate for p in groups(probe)
        $observation
    end)
    loop = plate.args[end]
    loop.args[2] = Expr(:block, body..., observation)
    response = direct ? :(y ~ Normal(mu, .7)) : plate
    source = Expr(:block, extra..., predictor, population, response)
    backend = RKBRMI(BRM.brm(source; mod=@__MODULE__)(data))
    (; backend, view=brm_nlme_view(backend))
end

@testset "GaussianBlockLaplace structural certificate" begin
    fixture = gaussian_certificate_fixture()
    c = BRM._brm_svi_gaussian_certificate(fixture.backend, fixture.view)
    @test length(c.globals) == 3
    @test length(c.field) == 8
    @test length(c.latent) == 12
    @test sort(vcat(c.globals, c.latent)) == collect(eachindex(fixture.view.coordinates))
    @test isempty(intersect(c.globals, c.latent))
    @test c.unit_blocks == fixture.view.subject_coordinates
    @test all(iszero, c.prior_mean)
    @test all(==(1.), c.prior_precision)
    @test c.scope.integration === :joint_conditional_gaussian
    @test !c.scope.streaming

    direct = gaussian_certificate_fixture(direct=true)
    cd = BRM._brm_svi_gaussian_certificate(direct.backend,direct.view)
    @test length(cd.globals)==3 && length(cd.latent)==12
    @test all(==(1.),cd.prior_precision)

    custom = gaussian_certificate_fixture(population=:(effect(mu, Intercept) ~ Normal(.4, 1.5)))
    cc = BRM._brm_svi_gaussian_certificate(custom.backend, custom.view)
    population = only(findall(==(:population), custom.view.role))
    k = only(findall(==(population), cc.latent))
    @test cc.prior_mean[k] == .4
    @test cc.prior_precision[k] ≈ inv(1.5^2)

    sampled_scale = gaussian_certificate_fixture(extra=[:(noise ~ Exponential(1))],
        observation=:(y[p] ~ Normal(mu[p], noise)))
    cs = BRM._brm_svi_gaussian_certificate(sampled_scale.backend, sampled_scale.view)
    @test length(cs.globals) == 4

    # A plate-local input takes precedence over a same-named scalar global.
    # Bind that input to the latent predictor: treating it as the global
    # noise parameter would incorrectly certify a latent-dependent scale.
    kernel = only(sampled_scale.backend.plan.assignments)
    local_scale = only(kernel.observations).arguments[2]
    @test local_scale === :noise && !(local_scale in kernel.params)
    shadowed = BRM._RKPreparedKernelAssignment((
        name === :params ? vcat(kernel.params, [local_scale]) :
        name === :inputs ? (kernel.inputs..., first(kernel.inputs)) :
        name === :globals ? filter(!=(local_scale), kernel.globals) :
        getfield(kernel, name) for name in fieldnames(typeof(kernel)))...)
    @test_throws ArgumentError BRM._brm_svi_gaussian_kernel(shadowed, Set([:mu]), Set([:noise]))

    # These are valid general BRM models, but fail the declared Gaussian
    # certificate: nonlinear means/scales or non-Gaussian collapsed priors
    # invalidate exact conditional integration and its prior spectral bound.
    variants = [
        (; observation=:(y[p] ~ Normal(exp.(mu[p]), .7))),
        (; observation=:(y[p] ~ Normal(mu[p].^2, .7))),
        (; observation=:(y[p] ~ Normal(mu[p], exp.(mu[p])))),
        (; population=:(effect(mu, Intercept) ~ Cauchy(0, 1))),
        (; linked=true),
    ]
    for options in variants
        f = gaussian_certificate_fixture(; options...)
        @test_throws ArgumentError BRM._brm_svi_gaussian_certificate(f.backend, f.view)
    end

    # An arbitrary cell assignment has no certificate, even if a numerical
    # spot check would happen to agree with an affine function.
    f = gaussian_certificate_fixture(body=[:(v = mu[p])], observation=:(y[p] ~ Normal(v, .7)))
    @test_throws ArgumentError BRM._brm_svi_gaussian_certificate(f.backend, f.view)
end
