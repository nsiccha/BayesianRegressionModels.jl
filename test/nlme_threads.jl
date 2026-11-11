# test/nlme_threads.jl — one-unit NLME calls from several tasks at once
# (src/nlme_view.jl, "Concurrent one-unit calls"): each call takes its own
# evaluator from the model's pool, so NLMEEstimation's
# `WarmupHMCNUTS(threads = true)` runs subjects in parallel.
#
# - overlapping calls on flat, plate and nested models equal serial calls bit
#   for bit, and the pool prepared more than one evaluator;
# - the pool prepares at most one evaluator per Julia thread, also when tasks
#   holding one yield;
# - `check_protocol(...; threads = true)` passes on a nested model;
# - a `threads = true` SAEM fit is the same bit for bit in processes with 1, 4
#   and N Julia threads (N from `BRM_NLME_THREADS_N`, default 8);
# - a `threads = true` fit agrees statistically with the lockstep fit
#   (`threads = false`) across seeds. Bit identity between the two is not
#   expected: BRM's batched and one-unit routes can differ in the last bits,
#   and NLMEEstimation 0.2.8 makes the routes equal only when they agree bit
#   for bit (the reproducibility contract).
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md, with real
# threads (it refuses to run without them):
#
#     julia --project=<scratch-env> --threads=4 test/nlme_threads.jl [testset-filter...]

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using NLMEEstimation, LinearAlgebra, Random, Statistics
const NE = NLMEEstimation

# 24 subjects of the plate PK model, simulated from known population values.
const THREADS_PK_TRUTH = (; log_CL=log(2.0), weight=0.5, log_V=log(10.0),
    omega_sd=(0.25, 0.2), sigma=0.4)

function threads_pk_data(; n=24, rng=Xoshiro(20261105))
    T = THREADS_PK_TRUTH
    times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
    weight = round.(0.3 .* randn(rng, n); digits=3)
    dose = rand(rng, [80.0, 100.0, 120.0], n)
    t = [sort!(times[randperm(rng, length(times))[1:rand(rng, 3:6)]]) for _ in 1:n]
    dv = map(1:n) do i
        CL = exp(T.log_CL + T.weight * weight[i] + T.omega_sd[1] * randn(rng))
        V = exp(T.log_V + T.omega_sd[2] * randn(rng))
        pred = dose[i] / V .* exp.(-(CL / V) .* t[i])
        round.(pred .+ T.sigma .* randn(rng, length(t[i])); digits=4)
    end
    (; subject=["s$i" for i in 1:n], dose, weight, t, dv)
end

threads_pk_model() = brm_nlme_model(RKBRMI(nlme_plate_pk(threads_pk_data()));
    ad_backend=NLME_AD)

const THREADS_SAEM = SAEMOptions(iterations=60, burnin=30, mcmc_burnin=5)

function threads_fit(m, seed; threads)
    s = brm_nlme_start(m)
    saem(m, s.theta, s.sigma, s.omega; rng=Xoshiro(seed),
        kernel=WarmupHMCNUTS(; threads), options=THREADS_SAEM)
end

fit_bits(fit) = reinterpret(UInt64, vcat(fit.theta, fit.sigma, fit.omega, vec(fit.eta)))

# A child process of the testset "the same in processes with 1, 4 and N
# threads": one seeded fit, reported as the hash of its exact bits.
if get(ENV, "BRM_NLME_THREADS_FIT", "") == "1"
    fit = threads_fit(threads_pk_model(), 2027; threads=true)
    println("NLME_THREADS_FIT threads=", Threads.nthreads(), " hash=",
        string(hash(fit_bits(fit)); base=16), " theta=", fit.theta,
        " sigma=", fit.sigma, " omega=", fit.omega)
    exit(0)
end

Threads.nthreads() >= 4 || error(
    "test/nlme_threads.jl needs real threads: run it with --threads=4")

julia_threads() = Threads.nthreads(:default) + Threads.nthreads(:interactive)

subject_eta(H::AbstractMatrix, i) = H[:, i]
subject_eta(H::AbstractVector, i) = H[i]

# Every one-unit route at every point and subject, once serially and many
# times from tasks on every thread, in shuffled order. Returns the number of
# evaluators the model's pool prepared.
function overlapping_calls_match(m, points)
    n = NE.nsubjects(m)
    calls(p, i) = let (θ, σ, H) = points[p], η = subject_eta(H, i)
        (NE.conditional_loglikelihood(m, i, θ, σ, η),
         NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η),
         NE.conditional_loglikelihood_and_population_gradient(m, i, θ, σ, η))
    end
    serial = Dict((p, i) => calls(p, i) for p in eachindex(points) for i in 1:n)
    jobs = shuffle!(Xoshiro(7), repeat(collect(keys(serial)), 30))
    results = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for k in eachindex(jobs)
        results[k] = calls(jobs[k]...)
    end
    @test all(isequal(results[k], serial[jobs[k]]) for k in eachindex(jobs))
    m.cell_gradient.evaluators.count[]
end

@stestset "overlapping one-unit calls equal serial calls bit for bit" begin
    rng = Xoshiro(11)
    models = (
        flat=brm_nlme_model(RKBRMI(nlme_pk(NLME_DATA)); ad_backend=NLME_AD),
        plate=threads_pk_model(),
        nested=brm_nlme_model(RKBRMI(nested_model(NESTED_DATA)); ad_backend=NLME_AD,
            subject_group=:subject, nested_groups=(unit=:obs_subject,)))
    for (name, m) in pairs(models)
        @test m.cell_gradient !== nothing
        s = brm_nlme_start(m)
        perturbed(H) = H isa AbstractMatrix ? H .+ 0.3 .* randn(rng, size(H)) :
            [h .+ 0.3 .* randn(rng, length(h)) for h in H]
        points = [(s.theta, s.sigma, s.eta),
            (s.theta .+ 0.1 .* randn(rng, length(s.theta)), s.sigma, perturbed(s.eta))]
        count = overlapping_calls_match(m, points)
        # More than one evaluator, or the calls never overlapped and this
        # testset checked nothing about concurrency.
        @test 1 < count <= julia_threads()
        @info "evaluators prepared" name count
    end
end

# A task that yields while it holds an evaluator lets its thread start another
# task, so holders can outnumber threads. Further tasks wait for an evaluator
# rather than prepare one, and every task finishes. `maxthreadid` is not the
# bound: it also counts the GC threads, so these tasks took the pool to 5
# evaluators at 4 threads before the pool was held to one per thread.
@stestset "the pool prepares at most one evaluator per Julia thread" begin
    m = threads_pk_model()
    pool = m.cell_gradient.evaluators
    limit = julia_threads()
    @test Threads.maxthreadid() > limit  # or this cannot tell the two bounds apart
    held, most = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
    @sync for _ in 1:8limit
        Threads.@spawn BRM._brm_nlme_with_evaluator(m) do _
            Threads.atomic_max!(most, Threads.atomic_add!(held, 1) + 1)
            sleep(0.05)
            Threads.atomic_sub!(held, 1)
        end
    end
    @test 1 < most[] <= limit
    @test 1 < pool.count[] <= limit
    @test length(pool.idle) == pool.count[]  # every holder gave its evaluator back
    # A pool that kept waiters waiting would have hung above; the pool still serves.
    s = brm_nlme_start(m)
    @test isfinite(first(NE.conditional_loglikelihood_and_gradient(m, 1, s.theta, s.sigma, s.eta[:, 1])))
end

@stestset "check_protocol with threads = true passes on a nested model" begin
    m = brm_nlme_model(RKBRMI(nested_model(NESTED_DATA)); ad_backend=NLME_AD,
        subject_group=:subject, nested_groups=(unit=:obs_subject,))
    s = brm_nlme_start(m)
    @test check_protocol(m, s.theta, s.sigma, s.omega; threads=true)
end

@stestset "a threads = true fit is the same in processes with 1, 4 and N threads" begin
    N = parse(Int, get(ENV, "BRM_NLME_THREADS_N", "8"))
    reports = map((1, 4, N)) do k
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$k
            --project=$(Base.active_project()) $(@__FILE__)`
        out = read(addenv(cmd, "BRM_NLME_THREADS_FIT" => "1"), String)
        line = only(filter(startswith("NLME_THREADS_FIT"), split(out, '\n')))
        println(line)
        @test occursin("threads=$k ", line)
        match(r"hash=(\w+)", line).captures[1]
    end
    @test allequal(reports)
end

@stestset "a threads = true fit agrees with the lockstep fit across seeds" begin
    m = threads_pk_model()
    names = m.view.coordinates[vcat(m.theta, m.sigma)]
    estimates(threads) = map(1:6) do seed
        fit = threads_fit(m, 100 + seed; threads)
        vcat(fit.theta, fit.sigma, fit.omega)
    end
    lockstep, threaded = estimates(false), estimates(true)
    L, T = reduce(hcat, lockstep), reduce(hcat, threaded)
    for c in axes(L, 1)
        spread = std(L[c, :])
        # The threaded fits' mean differs from the lockstep fits' by less than
        # twice the lockstep fits' run-to-run spread; a slack of 1e-3 covers a
        # coordinate whose fits all agree to the last digits.
        @test abs(mean(T[c, :]) - mean(L[c, :])) <= 2spread + 1e-3
    end
    # Both recover the simulated population values.
    truth = Dict(Symbol("pop_log_CL.beta_pop.1") => THREADS_PK_TRUTH.log_CL,
        Symbol("pop_log_CL.beta_pop.2") => THREADS_PK_TRUTH.weight,
        :log_V_Intercept => THREADS_PK_TRUTH.log_V)
    for (c, name) in enumerate(names)
        haskey(truth, name) || continue
        @test abs(mean(L[c, :]) - truth[name]) < 0.15
        @test abs(mean(T[c, :]) - truth[name]) < 0.15
    end
    @info "lockstep vs threaded population estimates" names mean_lockstep=vec(mean(L; dims=2)) mean_threaded=vec(mean(T; dims=2)) sd_lockstep=vec(std(L; dims=2))
end
