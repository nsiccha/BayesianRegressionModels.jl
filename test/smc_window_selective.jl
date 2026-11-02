using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Serialization, LinearAlgebra, Statistics
include("smc_window_fixture.jl")
include("testset_filter.jl")
const F = SMCWindowFixture

hierarchical_window(data) = @brm data begin
    m ~ Normal(0, 4)
    z[1:nb] .~ Normal.(m, 2)
    mu = z[block]
    y ~ Normal(mu, 1)
end

@stestset "An old-only window selects the likelihood without replaying old priors" begin
    training = (; nb=2, block=[1, 2], y=[0.0, 0.0])
    rk = RKBRMI(hierarchical_window(training))
    names = coordinate_names(rk.model.layout)
    moments = Dict(:m => (0.0, 1.0), Symbol("z.1") => (0.0, 1.0),
        Symbol("z.2") => (0.5, 0.9))
    carried = brm_smc_carry(rk; reference=SMCReference(
        [moments[name][1] for name in names], [moments[name][2] for name in names]))
    window = (; nb=2, block=[2], y=[1.5])
    options = SMCOptions(; moves=3, stepsize=0.25, max_tree_depth=4, history_limit=4)
    fit = brm_smc_update(carried, hierarchical_window(window);
        coordinates=:auto, rng=Xoshiro(1601), particles=1024, options)
    # m is a parent of z[2]'s old authored prior, which the carry replaces.
    @test size(fit.state.positions) == (1, 1024)
    @test only(F.weighted_mean(fit.state)) ≈ 0.5 + 0.81 / 1.81 atol=0.08
    @test only(F.weighted_variance(fit.state)) ≈ 0.81 / 1.81 atol=0.08
    @test fit.state.log_normalizer ≈ logpdf(Normal(0.5, sqrt(1.81)), 1.5) atol=0.15
end

@stestset "Selective particle storage does not multiply the inactive frame" begin
    # Large enough that a hidden full-width array costs over 17 MB when
    # increasing 32 particles to 256; compact per-particle bookkeeping is
    # allowed 4 KiB per added particle.
    training = (; nb=10_000, block=[1], y=[0.0])
    rk = RKBRMI(hierarchical_window(training))
    d = length(coordinate_names(rk.model.layout))
    carried = brm_smc_carry(rk; reference=SMCReference(zeros(d), ones(d)))
    window = (; nb=10_001, block=[10_001], y=[0.5])
    options = SMCOptions(; moves=1, stepsize=0.2, max_tree_depth=2, history_limit=2)
    fits = map((32, 256)) do particles
        brm_smc_update(carried, hierarchical_window(window);
            coordinates=:auto, rng=Xoshiro(1801), particles, options, stages=0)
    end
    @test size(fits[1].state.positions) == (2, 32)
    @test size(fits[2].state.positions) == (2, 256)
    state_bytes = Base.summarysize.((fits[1].state, fits[2].state))
    checkpoint_bytes = map(fit -> F.serialized_size(brm_smc_checkpoint(fit)), fits)
    println("BRM_SELECTIVE_STORAGE full_dimension=", d + 1,
        " active_dimension=2 particles=(32,256) state_bytes=", state_bytes,
        " checkpoint_bytes=", checkpoint_bytes)
    @test state_bytes[2] - state_bytes[1] < 4096 * (256 - 32)
    @test checkpoint_bytes[2] - checkpoint_bytes[1] < 4096 * (256 - 32)
end

@stestset "Selective windows include fresh-prior parents and preserve independent inactive coordinates" begin
    training = (; nb=2, block=[1, 2], y=[0.1, -0.2])
    rk = RKBRMI(hierarchical_window(training))
    names = coordinate_names(rk.model.layout)
    moments = Dict(:m => (1.0, 0.5), Symbol("z.1") => (-0.3, 0.7),
        Symbol("z.2") => (0.4, 0.9))
    carried = brm_smc_carry(rk; reference=SMCReference(
        [moments[name][1] for name in names], [moments[name][2] for name in names]))
    before = F.roundtrip(carried)
    fill!(training.y, NaN)
    window = (; nb=3, block=[3], y=[3.0])
    options = SMCOptions(; moves=3, stepsize=0.25, max_tree_depth=4, history_limit=4)
    rng = Xoshiro(1701)
    rng_before = copy(rng)
    partial = brm_smc_update(carried, hierarchical_window(window);
        coordinates=:auto, rng, particles=2048, options, stages=0)
    @test rng == rng_before
    @test F.equal(carried, before)
    # Only m (needed by the fresh prior) and z[3] have particle rows.
    @test size(partial.state.positions) == (2, 2048)
    @test length(coordinate_names(partial.backend.model.layout)) == 4
    @test partial.coordinates === :auto
    fit = brm_smc_fit(partial)
    @test fit.state.temperature == 1
    @test size(fit.state.positions) == (2, 2048)
    @test fit.state.log_normalizer ≈ logpdf(Normal(1, sqrt(5.25)), 3) atol=0.15

    next = brm_smc_carry(fit; basis=:fit, max_reflections=4, maxiter=100)
    new_names = coordinate_names(fit.backend.model.layout)
    location = Dict(zip(new_names, next.posterior.reference.location))
    @test location[:m] ≈ 1 + 0.25 / 5.25 * 2 atol=0.06
    @test location[Symbol("z.3")] ≈ 1 + 4.25 / 5.25 * 2 atol=0.08
    # Independent omitted coordinates keep their marginal exactly; the
    # old z priors must not be counted again under their authored m parent.
    @test location[Symbol("z.1")] ≈ -0.3 atol=1e-12
    @test location[Symbol("z.2")] ≈ 0.4 atol=1e-12
    @test F.equal(carried, before)

    checkpoint = F.roundtrip(brm_smc_checkpoint(partial))
    @test checkpoint.coordinates === :auto
    restored = brm_smc_fit(brm_smc_resume(hierarchical_window(window), checkpoint))
    @test F.equal(brm_smc_checkpoint(fit), brm_smc_checkpoint(restored))
    changed = merge(window, (; y=[4.0]))
    @test_throws r"window data has changed" brm_smc_resume(
        hierarchical_window(changed), checkpoint)
    @test_throws r"changes the fitted coordinates" brm_smc_update(
        carried, hierarchical_window(window); stages=0)
end

@stestset "Lazy recovery retains the correlated inactive conditional" begin
    training = (; nb=2, block=[1, 2], y=[0.0, 0.0])
    rk = RKBRMI(hierarchical_window(training))
    names = coordinate_names(rk.model.layout)
    order = Int.(indexin([:m, Symbol("z.1"), Symbol("z.2")], names))
    # Correlation .8 between m and the otherwise inactive z[1]; z[2]
    # is independent. Only this three-dimensional oracle uses a full draw.
    diagonal = ones(3)
    diagonal[order[1]] = sqrt(1.8)
    diagonal[order[2]] = sqrt(0.2)
    scale = (; diagonal, indices=[order[1:2]],
        reflections=[[cos(pi/8), sin(pi/8)]])
    carried = brm_smc_carry(rk; reference=SMCReference(zeros(3), scale))
    window = (; nb=3, block=[3], y=[2.0])
    fit = brm_smc_update(carried, hierarchical_window(window);
        coordinates=:auto, rng=Xoshiro(1901), particles=16, stages=0)
    @test size(fit.state.positions) == (2, 16)
    before = F.roundtrip(brm_smc_checkpoint(fit))
    compact = fill(1.25, 2)
    # Both selected values are equal, so this oracle is independent of
    # the compact ordering. The full joint uses current-model order.
    draws = reduce(hcat, (brm_smc_recover(fit, compact; rng=Xoshiro(2000 + i))
        for i in 1:1024))
    full_names = coordinate_names(fit.backend.model.layout)
    mi, z1i, z2i, z3i = Int.(indexin(
        [:m, Symbol("z.1"), Symbol("z.2"), Symbol("z.3")], full_names))
    @test size(draws) == (4, 1024)
    @test all(==(1.25), draws[mi, :])
    @test all(==(1.25), draws[z3i, :])
    @test mean(draws[z1i, :]) ≈ 0.8 * 1.25 atol=0.08
    @test var(draws[z1i, :]) ≈ 1 - 0.8^2 atol=0.06
    @test mean(draws[z2i, :]) ≈ 0 atol=0.12
    @test var(draws[z2i, :]) ≈ 1 atol=0.15
    @test compact == fill(1.25, 2)
    @test F.equal(brm_smc_checkpoint(fit), before)
end

positive_window(data) = @brm data begin
    z[1:nb] .~ Exponential(2)
    mu = z[block]
    y ~ Normal(mu, 1)
end

@stestset "Fresh positive coordinates retain their prior and transform Jacobian" begin
    training = (; nb=1, block=[1], y=[0.0])
    rk = RKBRMI(positive_window(training))
    carried = brm_smc_carry(rk; reference=SMCReference([0.7], [0.4]))
    window = (; nb=2, block=[2], y=[2.0])
    options = SMCOptions(; moves=3, stepsize=0.25, max_tree_depth=4, history_limit=4)
    fit = brm_smc_update(carried, positive_window(window);
        coordinates=:auto, rng=Xoshiro(2101), particles=2048, options)
    @test size(fit.state.positions) == (1, 2048)
    # Exp(scale=2) times Normal(y=2 | z, 1) is a Normal(1.5, 1)
    # restricted to z > 0. Missing the fresh log-Jacobian changes this law.
    oracle = truncated(Normal(1.5, 1), 0, Inf)
    z = exp.(vec(fit.state.positions))
    average = sum(fit.state.weights .* z)
    variance = sum(fit.state.weights .* (z .- average).^2)
    @test average ≈ mean(oracle) atol=0.08
    @test variance ≈ var(oracle) atol=0.10
    expected_logz = -log(2) - 1 + 1/8 + logcdf(Normal(), 1.5)
    @test fit.state.log_normalizer ≈ expected_logz atol=0.15
    next = brm_smc_carry(fit)
    names = coordinate_names(fit.backend.model.layout)
    @test next.posterior.reference.location[only(findall(==(Symbol("z.1")), names))] ≈
        0.7 atol=1e-12
end
