using Test
using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
using Distributions: BernoulliLogit, Exponential, LKJCholesky, Normal
using Enzyme
using LogDensityProblems
using Statistics: median
using StanBlocks

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL
const BS = BridgeStan

NP.@model function direct_shared_distributional_mixed(x, w, group)
    tau_p_group[(:mu_y, :log_sigma_y, :eta_z)] ~ Exponential(1)
    L_p_group[(:mu_y, :log_sigma_y, :eta_z)] ~ LKJCholesky(3, 2)
    b_p_group[group, (:mu_y, :log_sigma_y, :eta_z)] ~
        NP.MvNormalCholesky(tau_p_group, L_p_group)
    beta_mu_y[(:Intercept, :x)] ~ NP.StandardNormal()
    beta_log_sigma_y[(:Intercept, :w)] ~ NP.StandardNormal()
    mu_y = dot(beta_mu_y, (1, x)) +
        dot(b_p_group[group, (:mu_y,)], (1,))
    log_sigma_y = dot(beta_log_sigma_y, (1, w)) +
        dot(b_p_group[group, (:log_sigma_y,)], (1,))
    sigma_y = exp(log_sigma_y)
    @. y ~ Normal(mu_y, sigma_y)

    beta_eta_z[(:Intercept, :x)] ~ NP.StandardNormal()
    eta_z = dot(beta_eta_z, (1, x)) +
        dot(b_p_group[group, (:eta_z,)], (1,))
    @. z ~ BernoulliLogit(eta_z)
end

function realistic_mixed_data(; N=96, G=12)
    N % G == 0 || throw(ArgumentError("N must be divisible by G"))
    group = repeat(collect(1:G); inner=N ÷ G)
    x = collect(range(-1.8, 1.8; length=N))
    w = [cospi(2i / (N + 1)) for i in 1:N]
    group_mu = [0.18sinpi(2g / G) for g in 1:G]
    group_log_sigma = [0.10cospi(2g / G) for g in 1:G]
    group_eta = [0.22sinpi(4g / G) for g in 1:G]
    residual = repeat([-0.20, 0.08, -0.06, 0.14, -0.10, 0.04, 0.11, -0.03], G)
    y = [
        0.35 + 0.75x[i] + group_mu[group[i]] +
        exp(-0.45 + 0.12w[i] + group_log_sigma[group[i]]) * residual[i]
        for i in 1:N
    ]
    eta = [
        -0.15 + 0.65x[i] + group_eta[group[i]]
        for i in 1:N
    ]
    thresholds = repeat([-0.8, 0.5, -0.2, 0.9, -0.5, 0.2, -1.0, 0.7], G)
    z = Int[eta[i] > thresholds[i] for i in 1:N]
    (; x, w, group, y, z)
end

function native_position(G)
    vcat(
        log.([0.72, 0.48, 0.61]),
        [0.18, -0.12, 0.21],
        [0.16sin(0.7i) - 0.07cos(0.3i) for i in 1:(3G)],
        [0.32, 0.70, -0.48, 0.14, -0.18, 0.58],
    )
end

"""Map NativePPL's [log_tau, raw_L, z, beta] to Stan's [raw_L, log_tau, z, beta]."""
function stan_from_native(position, G)
    expected = 3 + 3 + 3G + 6
    length(position) == expected || throw(DimensionMismatch(
        "expected $expected NativePPL coordinates for G=$G; got $(length(position))"))
    vcat(position[4:6], position[1:3], position[7:end])
end

function native_from_stan_gradient(gradient, G)
    expected = 3 + 3 + 3G + 6
    length(gradient) == expected || throw(DimensionMismatch(
        "expected $expected Stan coordinates for G=$G; got $(length(gradient))"))
    vcat(gradient[4:6], gradient[1:3], gradient[7:end])
end

function expected_stan_names(G)
    vcat(
        ["b_p_group_L.$i" for i in 1:3],
        ["b_p_group_tau.$i" for i in 1:3],
        ["b_p_group_z_flat.$i" for i in 1:(3G)],
        ["pop_mu_y_beta_pop.$i" for i in 1:2],
        ["pop_log_sigma_y_beta_pop.$i" for i in 1:2],
        ["pop_eta_z_beta_pop.$i" for i in 1:2],
    )
end

function benchmark_call(f; warmup=1_000, samples=31, batch=200)
    for _ in 1:warmup
        f()
    end
    elapsed = Vector{Float64}(undef, samples)
    for sample in 1:samples
        started = time_ns()
        for _ in 1:batch
            f()
        end
        elapsed[sample] = (time_ns() - started) / batch
    end
    f()
    GC.gc()
    allocated = @allocated begin
        for _ in 1:batch
            f()
        end
    end
    (; median_ns=median(elapsed), min_ns=minimum(elapsed),
       bytes_per_call=allocated / batch, warmup, samples, batch)
end

function show_benchmark(backend, path, result; N, G, dimension)
    println(
        "BRM_MIXED_BENCH backend=", backend,
        " path=", path,
        " median_ns=", result.median_ns,
        " min_ns=", result.min_ns,
        " bytes_per_call=", result.bytes_per_call,
        " warmup=", result.warmup,
        " samples=", result.samples,
        " batch=", result.batch,
        " N=", N,
        " G=", G,
        " dimension=", dimension,
    )
end

@testset "unchanged BRMI parity across NativePPL, SBBRMI, and optimized Stan" begin
    N = 96
    G = 12
    data = realistic_mixed_data(; N, G)
    brmi = @brm data begin
        mu_y ~ 1 + x + (1 | p | group)
        log_sigma_y ~ 1 + w + (1 | p | group)
        eta_z ~ 1 + x + (1 | p | group)
        sd(:, p) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(3, 2)
        y ~ Normal(mu_y, exp(log_sigma_y))
        z ~ BernoulliLogit(eta_z)
    end

    direct = direct_shared_distributional_mixed(data.x, data.w, data.group)
    declaration = NP.lower(brmi)
    @test direct.declaration == declaration
    direct_plan = NP.compile(NP.condition(direct; y=data.y, z=data.z))
    brm_plan = NP.compile(brmi)
    @test direct_plan.graph.schedule == brm_plan.graph.schedule
    @test direct_plan.graph.coordinates == brm_plan.graph.coordinates
    @test brm_plan.graph.dimension == 48
    @test brm_plan.graph.coordinates.tau_p_group.indices == 1:3
    @test brm_plan.graph.coordinates.L_p_group.indices == 4:6
    @test brm_plan.graph.coordinates.b_p_group.indices == 7:42
    @test brm_plan.graph.coordinates.beta_mu_y.indices == 43:44
    @test brm_plan.graph.coordinates.beta_log_sigma_y.indices == 45:46
    @test brm_plan.graph.coordinates.beta_eta_z.indices == 47:48

    prepared = NP.prepare(brm_plan)
    direct_prepared = NP.prepare(direct_plan)
    dimension = LogDensityProblems.dimension(prepared)
    native_q = native_position(G)
    stan_q = stan_from_native(native_q, G)
    @test dimension == length(native_q)
    @test_throws DimensionMismatch stan_from_native(native_q[1:end-1], G)
    @test_throws DimensionMismatch native_from_stan_gradient(stan_q[1:end-1], G)

    native_work = NP.workspace(prepared, Float64, DI.AutoEnzyme())
    direct_work = NP.workspace(direct_prepared, Float64, DI.AutoEnzyme())
    native_lp, native_gradient = NP.logdensity_and_gradient!(
        native_work, prepared, native_q)
    direct_lp, direct_gradient = NP.logdensity_and_gradient!(
        direct_work, direct_prepared, native_q)
    @test direct_lp == native_lp
    @test direct_gradient == native_gradient

    sb = SBBRMI(brmi; mod=@__MODULE__)
    sbb_code = BRM.stan_code(sb)
    @test occursin("cholesky_factor_corr[n_terms_p_group] b_p_group_L", sbb_code)
    @test occursin("y ~ normal(mu_y, exp(log_sigma_y))", sbb_code)
    @test occursin("z ~ bernoulli_logit(eta_z)", sbb_code)
    sbb_path = joinpath(
        tempdir(), "brm-native-shared-sbb-$(UInt(hash(sbb_code))).stan")
    sbb_problem = StanBlocks.stan_instantiate(sb.model; path=sbb_path)

    optimized_source = joinpath(
        @__DIR__, "native_ppl_shared_distributional_mixed.stan")
    optimized_code = read(optimized_source, String)
    @test occursin("matrix[G, 3] b_p_group", optimized_code)
    @test occursin("matrix[N, 2] X_x", optimized_code)
    @test !occursin("generated quantities", optimized_code)
    optimized_path = joinpath(
        tempdir(), "brm-native-shared-optimized-$(UInt(hash(optimized_code))).stan")
    if !isfile(optimized_path)
        write(optimized_path, optimized_code)
    end
    optimized_data = StanBlocks.bridgestan_data(Dict(
        "N" => N, "G" => G, "x" => data.x, "w" => data.w,
        "group" => data.group, "y" => data.y, "z" => data.z,
    ))
    optimized_problem = StanBlocks.StanLogDensityProblems.StanProblem(
        optimized_path, optimized_data;
        nan_on_error=false, make_args=["STAN_THREADS=true"], warn=false)

    expected_names = expected_stan_names(G)
    sbb_names = BS.param_unc_names(sbb_problem.model)
    optimized_names = BS.param_unc_names(optimized_problem.model)
    @test sbb_names == expected_names
    @test optimized_names == expected_names
    @test sbb_names == optimized_names

    sbb_gradient = zeros(dimension)
    optimized_gradient = zeros(dimension)
    sbb_lp, _ = BS.log_density_gradient!(
        sbb_problem.model, stan_q, sbb_gradient;
        propto=false, jacobian=true)
    optimized_lp, _ = BS.log_density_gradient!(
        optimized_problem.model, stan_q, optimized_gradient;
        propto=false, jacobian=true)
    @test native_lp ≈ sbb_lp rtol=2e-12 atol=2e-10
    @test native_lp ≈ optimized_lp rtol=2e-12 atol=2e-10
    @test sbb_lp ≈ optimized_lp rtol=2e-12 atol=2e-10
    @test native_gradient ≈ native_from_stan_gradient(sbb_gradient, G) rtol=3e-11 atol=3e-10
    @test native_gradient ≈ native_from_stan_gradient(optimized_gradient, G) rtol=3e-11 atol=3e-10
    @test sbb_gradient ≈ optimized_gradient rtol=3e-11 atol=3e-10
    println(
        "BRM_MIXED_PARITY point=base",
        " native_lp=", native_lp,
        " sbb_lp=", sbb_lp,
        " optimized_lp=", optimized_lp,
        " max_native_sbb_gradient_diff=",
        maximum(abs, native_gradient .-
            native_from_stan_gradient(sbb_gradient, G)),
        " max_native_optimized_gradient_diff=",
        maximum(abs, native_gradient .-
            native_from_stan_gradient(optimized_gradient, G)),
    )

    shifted_native_q = native_q .+ 0.035 .* sin.(eachindex(native_q))
    shifted_stan_q = stan_from_native(shifted_native_q, G)
    shifted_native_lp, shifted_native_gradient = NP.logdensity_and_gradient!(
        native_work, prepared, shifted_native_q)
    shifted_sbb_lp, _ = BS.log_density_gradient!(
        sbb_problem.model, shifted_stan_q, sbb_gradient;
        propto=false, jacobian=true)
    shifted_optimized_lp, _ = BS.log_density_gradient!(
        optimized_problem.model, shifted_stan_q, optimized_gradient;
        propto=false, jacobian=true)
    @test shifted_native_lp ≈ shifted_sbb_lp rtol=2e-12 atol=2e-10
    @test shifted_native_lp ≈ shifted_optimized_lp rtol=2e-12 atol=2e-10
    @test shifted_native_gradient ≈
          native_from_stan_gradient(sbb_gradient, G) rtol=3e-11 atol=3e-10
    @test shifted_native_gradient ≈
          native_from_stan_gradient(optimized_gradient, G) rtol=3e-11 atol=3e-10
    println(
        "BRM_MIXED_PARITY point=shifted",
        " native_lp=", shifted_native_lp,
        " sbb_lp=", shifted_sbb_lp,
        " optimized_lp=", shifted_optimized_lp,
        " max_native_sbb_gradient_diff=",
        maximum(abs, shifted_native_gradient .-
            native_from_stan_gradient(sbb_gradient, G)),
        " max_native_optimized_gradient_diff=",
        maximum(abs, shifted_native_gradient .-
            native_from_stan_gradient(optimized_gradient, G)),
    )

    # The map is a permutation only: the explicit named-coordinate Jacobian is 1.
    @test stan_from_native(collect(1:dimension), G) ==
          vcat(4:6, 1:3, 7:dimension)
    @test 0.0 == log(1.0)

    NP.logdensity!(native_work, prepared, native_q)
    NP.logdensity_and_gradient!(native_work, prepared, native_q)
    native_density_bench = benchmark_call(
        () -> NP.logdensity!(native_work, prepared, native_q))
    native_gradient_bench = benchmark_call(
        () -> NP.logdensity_and_gradient!(native_work, prepared, native_q))
    sbb_density_bench = benchmark_call(
        () -> BS.log_density(
            sbb_problem.model, stan_q; propto=false, jacobian=true))
    sbb_gradient_bench = benchmark_call(
        () -> BS.log_density_gradient!(
            sbb_problem.model, stan_q, sbb_gradient;
            propto=false, jacobian=true))
    optimized_density_bench = benchmark_call(
        () -> BS.log_density(
            optimized_problem.model, stan_q; propto=false, jacobian=true))
    optimized_gradient_bench = benchmark_call(
        () -> BS.log_density_gradient!(
            optimized_problem.model, stan_q, optimized_gradient;
            propto=false, jacobian=true))

    @test native_density_bench.bytes_per_call == 0
    @test native_gradient_bench.bytes_per_call == 0
    @test all(isfinite, (
        native_density_bench.median_ns, native_gradient_bench.median_ns,
        sbb_density_bench.median_ns, sbb_gradient_bench.median_ns,
        optimized_density_bench.median_ns, optimized_gradient_bench.median_ns,
    ))
    show_benchmark("NativePPL", "density!", native_density_bench; N, G, dimension)
    show_benchmark("NativePPL", "DI.AutoEnzyme density+gradient!", native_gradient_bench; N, G, dimension)
    show_benchmark("SBBRMI/BridgeStan", "normalized density FFI", sbb_density_bench; N, G, dimension)
    show_benchmark("SBBRMI/BridgeStan", "normalized gradient FFI", sbb_gradient_bench; N, G, dimension)
    show_benchmark("optimized Stan/BridgeStan", "normalized density FFI", optimized_density_bench; N, G, dimension)
    show_benchmark("optimized Stan/BridgeStan", "normalized gradient FFI", optimized_gradient_bench; N, G, dimension)
end
