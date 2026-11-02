using Test, Statistics, Treebars
include("smc_laplace_fixture.jl")
include("testset_filter.jl")
const SMCEXT = Base.get_extension(BRM, :BayesianRegressionModelsSMCExt)

@testset "Prepared Laplace SMC" begin

@stestset "Laplace SMC exact Gaussian marginals and weighted joint completion" begin
    for (kind, expected_type) in ((:generic, BRMLaplaceProblem),
            (:structured, BRMStructuredLaplaceProblem), (:sparse, BRMSparseLaplaceProblem))
        @testset "$kind" begin
        p = lsmc_problem(kind)
        @test p isa expected_type
        rk = SMCEXT.laplace_backend(p)
        oracle = lsmc_oracle(rk)
        higher = lsmc_oracle(rk; quadrature=96)
        @test oracle.mean ≈ higher.mean atol=1e-10
        @test oracle.covariance ≈ higher.covariance atol=1e-10
        @test oracle.logz ≈ higher.logz atol=1e-10
        for x in (-0.8, 0.0, 0.6)
            phi = fill(x, length(p.population))
            v, g = LogDensityProblems.logdensity_and_gradient(p, phi)
            @test v ≈ oracle.marginal(p.population, phi) atol=2e-9
            numerical = map(eachindex(phi)) do j
                plus = copy(phi); minus = copy(phi)
                plus[j] += 1e-5; minus[j] -= 1e-5
                (oracle.marginal(p.population, plus) - oracle.marginal(p.population, minus)) / 2e-5
            end
            @test g ≈ numerical atol=2e-8
        end
        original = (copy(p.u), p.nonfinite, p.unconverged)
        rng = Xoshiro(57); expected_rng = copy(rng)
        initial = brm_smc_fit(p; rng, particles=1536, options=LSMC_OPTIONS, stages=0)
        @test rand(rng) == rand(expected_rng)
        @test original == (p.u, p.nonfinite, p.unconverged)
        @test initial.backend !== p && initial.backend.u !== p.u
        @test initial.state.temperature == 0
        @test_throws ArgumentError brm_laplace_draws(initial)
        cp = lsmc_roundtrip(brm_smc_checkpoint(initial))
        fit = brm_smc_fit(initial)
        replay = brm_smc_fit(brm_smc_resume(lsmc_problem(kind), cp))
        @test fit.state.positions == replay.state.positions
        @test fit.state.weights == replay.state.weights
        @test fit.state.log_normalizer == replay.state.log_normalizer
        @test original == (p.u, p.nonfinite, p.unconverged)
        @test initial.state.temperature == 0
        fits = [fit, brm_smc_fit(p; rng=Xoshiro(158), particles=1536, options=LSMC_OPTIONS)]
        comparison = brm_smc_compare(fits...)
        @test comparison.names == p.names
        for fit in fits
            @test fit.state.temperature == 1
            @test fit.state.log_normalizer ≈ oracle.logz atol=0.15
            m = fit.state.positions * fit.state.weights
            v = (fit.state.positions .- m).^2 * fit.state.weights
            @test m ≈ oracle.mean[p.population] atol=0.08
            @test v ≈ diag(oracle.covariance)[p.population] atol=0.08
            for conditional in (0, 8)
                rng = Xoshiro(27); expected_rng = copy(rng)
                out = brm_laplace_draws(fit; rng, conditional)
                @test rand(rng) == rand(expected_rng)
                @test out.weights == fit.state.weights && out.weights !== fit.state.weights
                @test out.draws[p.population, :] == fit.state.positions
                @test hasproperty(out.diagnostics, :importance_ess) == (conditional > 0)
                joint_mean = out.draws * out.weights
                centered = out.draws .- joint_mean
                covariance = (centered .* transpose(out.weights)) * centered'
                @test joint_mean ≈ oracle.mean atol=0.12
                @test covariance ≈ oracle.covariance atol=0.17
            end
        end
        stopped = Treebars.with_progress(:state; description="marginal interruption") do node
            Treebars.request_interrupt!(node)
            brm_smc_fit(initial; progress=node)
        end
        @test stopped.state.positions == initial.state.positions
        @test stopped.state.temperature == 0
        @test_throws ArgumentError brm_smc_carry(fit)
        end
    end
end

@stestset "Laplace SMC identities, pilots and cumulative updates" begin
    p = lsmc_problem()
    rng = Xoshiro(9); untouched_rng = copy(rng)
    init = [0.1]; untouched_init = copy(init)
    schedule = brm_smc_tune(p; rng, init, temperatures=[0.0, 0.5, 1.0],
        pilot_steps=12, options=LSMC_OPTIONS)
    @test rng == untouched_rng && init == untouched_init
    @test all(iszero, p.u)
    options = SMCOptions(; tuning=schedule, moves=2, max_tree_depth=4)
    initial = brm_smc_fit(p; rng=Xoshiro(23), particles=128, options, stages=0)
    checkpoint = lsmc_roundtrip(brm_smc_checkpoint(initial))
    expected = brm_smc_fit(initial)
    resumed = brm_smc_fit(brm_smc_resume(lsmc_problem(), checkpoint))
    @test expected.state.positions == resumed.state.positions
    @test expected.state.weights == resumed.state.weights
    @test expected.state.log_normalizer == resumed.state.log_normalizer
    changed = lsmc_problem(); changed.tol *= 10
    @test_throws ArgumentError brm_smc_resume(changed, checkpoint)
    @test_throws ArgumentError brm_smc_fit(changed; options, particles=8)
    changed = lsmc_problem(); changed.maxiter += 1
    @test_throws ArgumentError brm_smc_resume(changed, checkpoint)
    changed = lsmc_problem(); changed.names[1] = :changed
    @test_throws ArgumentError brm_smc_resume(changed, checkpoint)
    @test_throws ArgumentError brm_smc_resume(lsmc_problem(:sampled), checkpoint)
    data = (; g=repeat(["a", "b", "c"], inner=3),
        y=[0.3, 0.9, 0.5, -0.5, 0.1, 0.2, 1.1, 0.7, 0.6])
    rk = reprocess(p.backend, data)
    next = lsmc_problem(; rk)
    before = lsmc_roundtrip(brm_smc_checkpoint(expected))
    @test lsmc_equal(before, brm_smc_checkpoint(expected))
    changed_checkpoint = lsmc_roundtrip(before)
    changed_checkpoint.state.positions[1] += 1
    @test !lsmc_equal(before, changed_checkpoint)
    @test_throws ArgumentError brm_smc_update(expected, data)
    tuned = brm_smc_tune(expected, next; rng=Xoshiro(30), init=[0.0],
        temperatures=[0.0,1.0], pilot_steps=10, options=LSMC_OPTIONS)
    partial = brm_smc_update(expected, next;
        options=SMCOptions(; tuning=tuned, moves=2, max_tree_depth=4), stages=0)
    updated = brm_smc_fit(partial)
    restored = brm_smc_fit(brm_smc_resume(lsmc_problem(), brm_smc_checkpoint(partial); current=next))
    @test updated.state.positions == restored.state.positions
    @test updated.state.weights == restored.state.weights
    @test lsmc_equal(before, brm_smc_checkpoint(expected))
    @test_throws ArgumentError brm_smc_resume(p, brm_smc_checkpoint(partial))
    @test_throws ArgumentError brm_smc_compare(expected, updated)
    @test_throws ArgumentError brm_smc_update(expected, changed; options=LSMC_OPTIONS)
    # Changing fitted standardization must reject even when dimension is equal.
    scaled(d) = @brm d begin
        mu ~ 1 + zscale(x) + (1 | g)
        sd(mu, g) = 0.7
        y ~ Normal(mu, 0.8)
    end
    d1 = (; LSMC_DATA..., x=[1.,2.,3.,4.,5.,6.])
    d2 = (; d1..., x=2 .* d1.x)
    q1 = brm_laplace_problem(RKBRMI(scaled(d1)); ad_backend=LSMC_AD, integrate=(:g,))
    q2 = brm_laplace_problem(RKBRMI(scaled(d2)); ad_backend=LSMC_AD, integrate=(:g,))
    @test_throws ArgumentError SMCEXT.smc_update_model(q1, q2)
    @test SMCEXT.smc_update_model(q1, brm_laplace_problem(reprocess(q1.backend, d2);
        ad_backend=LSMC_AD, integrate=(:g,))) isa BRMSparseLaplaceProblem
    # SMC retains explicit models, so it must not impose fitted-state export's
    # stricter data-free boundary on CAR fields.
    spatial = (; area=["a","b","c","d"], from=["a","b","c"],
        to=["b","c","d"], y=[0.1,-0.2,0.6,0.3])
    carmodel = RKBRMI(@brm spatial begin
        mu ~ 1 + car(area; edges=(from,to), type=:icar)
        y ~ Normal(mu, 0.8)
    end)
    carp = brm_laplace_problem(carmodel; ad_backend=LSMC_AD, integrate=(:area,))
    carfit = brm_smc_fit(carp; particles=8, rng=Xoshiro(12), stages=0)
    @test carfit.state.temperature == 0
    @test brm_smc_resume(carp, brm_smc_checkpoint(carfit)).binding_id == carfit.binding_id
end

@stestset "Laplace SMC zero integration and non-Gaussian smoke" begin
    p = lsmc_problem(:sampled)
    @test isempty(p.integrated)
    fit = brm_smc_fit(p; rng=Xoshiro(51), particles=1024, options=LSMC_OPTIONS)
    out = brm_laplace_draws(fit; conditional=8)
    @test out.draws == fit.state.positions
    oracle = lsmc_oracle(p.backend)
    @test out.draws * out.weights ≈ oracle.mean atol=0.15
    @test fit.state.log_normalizer ≈ oracle.logz atol=0.2
    rk = RKBRMI(@brm (; g=LSMC_DATA.g, y=[1,2,0,1,3,2]) begin
        log(mu) ~ 1 + (1 | g)
        effect(mu, Intercept) ~ Normal(0, 1)
        sd(mu, g) = 0.5
        y ~ Poisson(mu)
    end)
    p = brm_laplace_problem(rk; ad_backend=LSMC_AD, integrate=(:g,))
    v, g = LogDensityProblems.logdensity_and_gradient(p, [0.2])
    fd = (LogDensityProblems.logdensity(p, [0.20001]) -
        LogDensityProblems.logdensity(p, [0.19999])) / 2e-5
    @test g[1] ≈ fd rtol=1e-6
    fit = brm_smc_fit(p; rng=Xoshiro(9), particles=64, options=LSMC_OPTIONS)
    @test fit.state.temperature == 1
    @test all(isfinite, brm_laplace_draws(fit; conditional=12).draws)
end

@stestset "Laplace SMC independent copies and threaded particles" begin
    isdefined(Enzyme.Compiler, :RestoreSafepointEffectsPass) || error("concurrent-GC repair required")
    for kind in (:generic, :structured, :sparse, :sampled)
        p = lsmc_problem(kind)
        LogDensityProblems.logdensity_and_gradient(p, zeros(length(p.population)))
        a = SMCEXT.copy_laplace(p); b = SMCEXT.copy_laplace(p)
        @test a.u !== b.u && a.u !== p.u
        @test a.population !== b.population
        kind === :sparse && @test a.factor[] === b.factor[] === nothing
        values = [LogDensityProblems.logdensity_and_gradient(q, fill(0.1, length(q.population)))
            for q in (a,b)]
        @test values[1] == values[2]
        if Threads.nthreads() > 1
            serial = brm_smc_fit(p; rng=Xoshiro(14), particles=64,
                options=SMCOptions(; moves=2, stepsize=0.4, max_tree_depth=3, nthreads=1))
            parallel = brm_smc_fit(p; rng=Xoshiro(14), particles=64,
                options=SMCOptions(; moves=2, stepsize=0.4, max_tree_depth=3, nthreads=2))
            @test serial.state.positions == parallel.state.positions
            @test serial.state.weights == parallel.state.weights
        end
    end
end

@stestset "Laplace SMC propagates nonfinite failures without changing inputs" begin
    # Valid finite inputs whose deliberately unstable sigmoid arithmetic
    # overflows only when the prepared density evaluates a particle.
    diagnostic_model(d) = @brm d begin
        mu ~ 1 + (1 | g)
        effect(mu, Intercept) ~ Normal(0, 1)
        sd(mu, g) = 0.7
        exploding = exp(x * mu^2)
        location = exploding / (1 + exploding)
        y ~ Normal(location, 1)
    end
    sampled(rk) = brm_laplace_problem(rk; ad_backend=LSMC_AD,
        integrate=brm_laplace_plan(rk; ad_backend=LSMC_AD,
            integrate=(), sample=(:g,), draws=8))
    data = (; g=["a", "b"], x=0.0, y=[0.3, 0.1])
    p = sampled(RKBRMI(diagnostic_model(data)))
    rng = Xoshiro(71); untouched = copy(rng)
    initial = brm_smc_fit(p; rng, particles=32, options=LSMC_OPTIONS, stages=0)
    fit = brm_smc_fit(initial)
    checkpoint = lsmc_roundtrip(brm_smc_checkpoint(fit))
    bad = sampled(reprocess(p.backend, (; data..., x=1e8)))
    for progress in (nothing, :state)
        @test_throws SMCNonfiniteError brm_smc_fit(bad; rng, particles=32, progress)
        @test_throws SMCNonfiniteError brm_smc_update(fit, bad; options=LSMC_OPTIONS, progress)
        @test lsmc_equal(checkpoint, brm_smc_checkpoint(fit))
    end
    @test rng == untouched
    @test all(iszero, p.u)
end

@stestset "Laplace SMC ladder uses weights and conditional completion" begin
    p = lsmc_problem()
    fit = brm_smc_fit(p; particles=128, rng=Xoshiro(90), options=LSMC_OPTIONS)
    saved = copy(fit.state.positions)
    sampled = BRM._brm_ladder_samples(fit, p, Xoshiro(6))
    @test sampled.method === :smc
    @test fit.state.positions == saved
    @test size(sampled.positions) == size(saved)
    # An exact degenerate weighting is a positive control against silently
    # treating the original weighted cloud as equal-weight columns.
    fit.state.weights .= 0; fit.state.weights[3] = 1
    sampled = BRM._brm_ladder_samples(fit, p, Xoshiro(6))
    @test all(sampled.positions[:,j] == saved[:,3] for j in axes(saved,2))
    sampler(q, init, rng) = brm_smc_fit(q; particles=32, rng, options=LSMC_OPTIONS)
    ladder = brm_ladder_fit(p.backend; ad_backend=LSMC_AD, upto=5,
        integrate=(:g,), sampler, draws=8, conditional=6, rng=Xoshiro(3))
    rung = only(x for x in ladder.ladder if x.name === :population_sampling)
    @test rung.method === :smc && rung.conditional == 6
    @test size(ladder.draws,2) == 32
end

@stestset "Laplace SMC fresh-process checkpoint replay" begin
    mktempdir() do dir
        path = joinpath(dir,"checkpoint.bin")
        output = joinpath(dir,"replayed.bin")
        p = lsmc_problem()
        schedule = brm_smc_tune(p; rng=Xoshiro(2), init=[0.0],
            temperatures=[0.0,1.0], pilot_steps=8, options=LSMC_OPTIONS)
        initial = brm_smc_fit(p; rng=Xoshiro(80), particles=32, stages=0,
            options=SMCOptions(; tuning=schedule, max_tree_depth=3))
        serialize(path, brm_smc_checkpoint(initial))
        expected = brm_smc_fit(initial)
        script = joinpath(@__DIR__,"smc_laplace_checkpoint_process.jl")
        run(`$(Base.julia_cmd()) --startup-file=no --threads=$(Threads.nthreads()) --project=$(@__DIR__) $script $path $output`)
        actual = deserialize(output)
        @test lsmc_mismatch(brm_smc_checkpoint(expected), actual) === nothing
    end
end

end # Prepared Laplace SMC
