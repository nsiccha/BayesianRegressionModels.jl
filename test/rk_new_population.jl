# Seeded new levels through the same vectorized reader a consumer prepares.
using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "testset_filter.jl"))
using BayesianRegressionModels, Distributions, ReactiveKernelsPPL
import ReactiveKernels as RK
const BRM = BayesianRegressionModels

gaussian = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
end
independent = @brm begin
    mu ~ 1 + x + (1 + x || subject)
    y ~ Normal(mu, 1.0)
end
student = @brm begin
    sigma ~ Exponential(1)
    nu ~ Gamma(2, 3)
    mu ~ 1 + x + (1 + x | p | subject)
    ranef(:, p) ~ TDist(nu)
    y ~ Normal(mu, sigma)
end
curves = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | subject) + hsgp(x; by=series, k=4, c=2.0)
    log(sd(hsgp(x))) ~ 1 + (1 | series)
    log(length_scale(hsgp(x))) ~ 1 + (1 | series)
    y ~ Normal(mu, sigma)
end
const training = (; subject=[1,1,2,2,3,3], series=[1,1,2,2,3,3],
    x=[0.,0.2,0.4,0.6,0.8,1.], y=[0.,1.,0.,1.,0.,1.])
const seed = 19

function read_mu(rk, binder, design, U, effects; schedule=(;))
    bound = binder(design)
    have = (:unconstrained, keys(effects)..., keys(bound)...)
    scalar = RK.prepare(rk.model.spec; have, want=:mu, bound)
    batch = RK.vectorize(scalar; batched=(:unconstrained, keys(effects)...), schedule...)
    result = Base.invokelatest(batch, U, values(effects)...)
    result isa Tuple ? only(result) : result
end

# An independent calculation given the same random innovations. Its transform
# uses constrained fit parameters and an explicit per-subject scale mixture.
function expected_effects(rk, u, n, d; heavy)
    c = constrain(rk.model.layout, u)
    function rng(name)
        key = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)._rk_population_key(seed, (:subject,), name)
        Xoshiro(UInt128(key) << 64 | UInt128(d))
    end
    z = rand(rng(Symbol("b_p_subject.z")), Normal(), n, 2)
    e = z * transpose(Diagonal(c.b_p_subject.tau) * c.b_p_subject.L)
    if heavy
        w = rand(rng(Symbol("b_p_subject.w")), InverseGamma(c.nu / 2, c.nu / 2), n)
        e .*= sqrt.(w)
    end
    e
end

@stestset "fitted Gaussian and Student-t laws at absolute draw positions" begin
    for (builder, heavy) in ((gaussian,false), (student,true))
        rk = RKBRMI(builder(training))
        binder = rk_design_binder(rk; resample_groups=(:subject,))
        U = randn(Xoshiro(31), rk.model.layout.total, 5) ./ 2
        positions = [1, 8, 9, 3, 5]
        for n in (2, 7)
            design = (; subject=collect(11:10+n), x=collect(range(0., 1.; length=n)), y=zeros(n))
            effects = rk_new_population_effects(binder, design, U; seed, draw_positions=positions)
            @test keys(effects) == keys(binder.level_inputs) == (:b_p_subject,)
            b = effects.b_p_subject
            @test size(b) == (n, 2, 5)
            for (j,d) in pairs(positions)
                @test b[:,:,j] ≈ expected_effects(rk, U[:,j], n, d; heavy)
            end
            changed = merge(design, (; x=reverse(design.x)))
            perm = [4,1,3]
            selected = rk_new_population_effects(binder, changed, U[:,perm]; seed,
                draw_positions=positions[perm])
            @test selected.b_p_subject == b[:,:,perm]
            @test rk_new_population_effects(binder, design, U; seed=seed+1,
                draw_positions=positions).b_p_subject != b
            zeros_b = (; b_p_subject=zeros(size(b)))
            population = read_mu(rk,binder,design,U,zeros_b)
            mu = read_mu(rk,binder,design,U,effects)
            @test mu ≈ population .+ b[:,1,:] .+ design.x .* b[:,2,:]
            @test mu ≈ read_mu(rk,binder,design,U,effects;
                schedule=(; schedule=RK.NativeScheduling(chunk_size=2)))
            @test size(rk_new_population_effects(binder,design,U[:,1:0];seed).b_p_subject) == (n,2,0)
            @test_throws DimensionMismatch rk_new_population_effects(binder,design,U;seed,draw_positions=[1])
            @test_throws ArgumentError rk_new_population_effects(binder,design,U;seed,draw_positions=0:4)
        end
    end
end

@stestset "independent Gaussian intercepts and slopes in the same grouping" begin
    rk = RKBRMI(independent(training))
    binder = rk_design_binder(rk; resample_groups=(:subject,))
    U = randn(Xoshiro(15), rk.model.layout.total, 2)
    design = (; subject=[8,9], x=[0.2,0.5], y=zeros(2))
    effects = rk_new_population_effects(binder, design, U; seed)
    @test keys(effects) == keys(binder.level_inputs)
    @test length(effects) == 2
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    for (name, effect) in pairs(effects)
        @test size(effect) == (2, 1, 2)
        key = ext._rk_population_key(seed, (:subject,), Symbol(name, ".z"))
        for j in axes(U, 2)
            scale = only(constrain(rk.model.layout, U[:, j])[name].tau)
            rng = Xoshiro(UInt128(key) << 64 | UInt128(j))
            @test effect[:, :, j] ≈ scale .* rand(rng, Normal(), 2, 1)
        end
    end
    @test all(isfinite, read_mu(rk, binder, design, U, effects))
end

@stestset "two groupings, standardized curve weights, scalar hyper effects and data-free fit" begin
    rk = RKBRMI(curves(training))
    U = randn(Xoshiro(49), rk.model.layout.total, 4) ./ 3
    d = (; subject=[21,22,23,24], series=[12,11,11,12], x=[0.2,0.7,0.9,0.1], y=zeros(4))
    groups = (:series,:subject)
    binder = rk_design_binder(rk; resample_groups=groups)
    e = rk_new_population_effects(binder,d,U;seed)
    @test keys(e) == keys(binder.level_inputs)
    @test size(e.b_subject) == (4,1,4)
    weights, scales = e[Symbol("hsgp_x.z")], e[Symbol("hsgp_x.sigma_z")]
    lengths = e[Symbol("hsgp_x.rho_z")]
    @test size(weights) == (2,4,4)
    @test size(scales) == (2,4)
    @test size(lengths) == (2,4)
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    for j in axes(U, 2)
        rng(group, name) = Xoshiro(UInt128(ext._rk_population_key(seed, (group,), name)) << 64 | UInt128(j))
        c = constrain(rk.model.layout, U[:, j])
        @test e.b_subject[:, :, j] ≈ c.b_subject.tau[1] .*
            rand(rng(:subject, Symbol("b_subject.z")), Normal(), 4, 1)
        @test weights[:, :, j] == rand(rng(:series, Symbol("hsgp_x.z")), Normal(), 2, 4)
        @test scales[:, j] == rand(rng(:series, Symbol("hsgp_x.sigma_z")), Normal(), 2)
        @test lengths[:, j] == rand(rng(:series, Symbol("hsgp_x.rho_z")), Normal(), 2)
    end
    tasks = map(([4,2], [1,3])) do J
        Threads.@spawn rk_new_population_effects(binder, d, U[:,J]; seed, draw_positions=J)
    end
    for (task, J) in zip(tasks, ([4,2], [1,3]))
        part = fetch(task)
        @test all(part[n] == selectdim(e[n], ndims(e[n]), J) for n in keys(e))
    end
    @test all(isfinite, read_mu(rk,binder,d,U,e))
    state = brm_fitted_state(rk)
    free = RKBRMI(curves(d), state; resample_groups=groups)
    @test rk_new_population_effects(rk_design_binder(free; resample_groups=groups),d,U;seed) == e
    @test rk_new_population_effects(rk_design_binder(rk),training,U;seed) == (;)
end
