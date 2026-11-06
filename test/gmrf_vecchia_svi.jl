# A read-pair gmrf field through brm_svi_fit: the partition's unit terms and the global
# term add up to the whole model, the public fit runs and predicts, and the mesh form
# is refused by name.
using Test, BayesianRegressionModels, LinearAlgebra, Random, Statistics
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface
const BRM = BayesianRegressionModels
const FIELD_AD = AutoEnzyme(; mode=Enzyme.Reverse)

function field_sites(sites, rows; seed=1)
    rng = Xoshiro(seed)
    px, py = rand(rng, sites), rand(rng, sites)
    site = repeat(1:sites; inner=rows)
    x = randn(rng, length(site))
    y = sin.(3 .* px[site]) .+ 0.3 .* randn(rng, sites)[site] .* x .+ 0.3 .* randn(rng, length(site))
    (; east=px[site], north=py[site], time=zeros(length(site)), x, y, site=string.("s", site),
       state=collect(eachindex(site)), nb=5)
end

function field_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; neighbors=nb)
        sd(field, gmrf((east, north), time)) ~ Exponential(1)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.3), 0.7)
        length_scale(field, gmrf((east, north), time), time) = 10.0
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
end

data = field_sites(8, 3)
rk = RKBRMI(field_model(data))
m = brm_nlme_model(rk; ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
names = String.(m.view.coordinates)
states = findall(n -> occursin(".field.", n), names)

@testset "the view reads the field as shared" begin
    @test length(states) == 8
    @test all(==(:shared_latent), m.view.role[states])
    @test all(iszero, m.view.subject[states])
end

@testset "unit terms and the global term add up to the whole model" begin
    t = BRM._brm_svi_target(m)
    @test states ⊆ t.globals
    sampler = prepare_sampler(rk.model, BRM._rk_translated(rk), m.unit; backend=FIELD_AD)
    for offset in (-0.2, 0.0, 0.25)
        u = [0.1sin(i) + offset for i in eachindex(m.unit)]
        g = u[t.globals]
        value, gg = BRM._brm_svi_global_term(t, g)
        grad = zeros(length(u))
        grad[t.globals] .= gg
        for (i, block) in enumerate(t.units)
            gi, gz = zeros(length(g)), zeros(length(block))
            value += BRM._brm_svi_unit_term!(gi, gz, t, i, g, u[block])
            grad[t.globals] .+= gi
            grad[block] .+= gz
        end
        expected_grad = similar(u)
        expected, _ = sampler_value_and_gradient!(sampler, expected_grad, u)
        @test value ≈ expected atol=2e-9 rtol=2e-11
        @test grad ≈ expected_grad atol=2e-8 rtol=2e-9
    end
end

@testset "the public fit runs and its draws predict" begin
    fit = brm_svi_fit(m; batch_size=4, max_iterations=20, ndraws=5, threads=1, rng=Xoshiro(2))
    @test fit.diagnostics.iterations == 20
    @test all(isfinite, fit.diagnostics.elbo)
    out = brm_gmrf_predict(rk, fit.draws, :field; east=[0.5, 0.2], north=[0.5, 0.8], time=[0.0, 0.0])
    @test size(out) == (2, 5) && all(isfinite, out)
    # At a read site the prediction is the term's own output in that draw.
    read = brm_gmrf_predict(rk, fit.draws, :field; east=data.east[1:1], north=data.north[1:1], time=[0.0])
    output = rk_draw_values(rk, fit.draws, (:field,)).field
    @test vec(read) ≈ [o[1] for o in output] rtol=1e-10
end

@testset "the mesh form is refused by name" begin
    xy = [0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5]
    triangles = [1 2 5; 2 3 5; 3 4 5; 4 1 5]
    input = merge(data, (; xy, triangles))
    mesh = @brm input begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
    mm = brm_nlme_model(RKBRMI(mesh); ad_backend=FIELD_AD, subject_group=:site, unit_partition=:connected)
    @test_throws BRM.BRMNLMEViewError BRM._brm_svi_target(mm)
    err = try BRM._brm_svi_target(mm); nothing catch e; e end
    @test occursin("neighbors=m", sprint(showerror, err))
end
