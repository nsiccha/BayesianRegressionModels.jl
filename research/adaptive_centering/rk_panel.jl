# The synthetic panel behind brm-use's figures for RKBRMI adaptive centring: the
# plain RK density against `adaptive_centering_problem(rk, problem, backend)`
# under NUTS through WarmupHMC's completion quorum, on an RKBRMI whose group
# intercepts the data pin, so the non-centred intercept block crawls along the
# ridge `level_g = beta + tau * z_g`.
# Snag rkbrmi-has-no-ce-3158a399 measured it for BRM 0.2.263; this file is that
# run's script (KB upload sha256 49c1384893ecb2d749da49505cb7fcf8cc3c8c3a2183988b01888b38775317ca),
# with only this header changed (snag reproduce-the-ex-0ebad05a).
#
# What defines the benchmark, all below:
# - data: `panel(; seed=1)`, 12 groups x 30 rows; level ~ 60 + 70 U(0, 1),
#   per-group slope deviations 0.5 N(0, 1) on t, c1, c2 around 1.0, 0.5, -0.3;
#   t a grid on [-1, 1]; c1, c2 standard normal per row; known per-row
#   standard error se ~ 0.3 + 0.3 U(0, 1); all drawn from `Xoshiro(1)` in the
#   order written. On Julia 1.10.12: sum(y) = 33313.685919, sum(c1) =
#   35.44997, sum(c2) = 19.590428, sum(se) = 162.991494, y[1] = 64.002679;
# - model and priors: `model`: the four `||` SDs as written, every population
#   coefficient (intercept included) BRM's default Normal(0, 1) (the RKPPL
#   program reads `pop_mu ~ brm_population_effects(X_mu, 4, 0.0, 1.0)`), every
#   standardized group effect Normal(0, 1), `y ~ Normal(mu, se)`;
# - sampling: `completion_warmup_mcmc` with WarmupHMC's defaults apart from
#   `min_completed=3, n_draws=400`; 4 chains with RNGs `Xoshiro(1000seed + i)`,
#   `i = 1:4`, for each seed (default 51 and 52); AutoEnzyme(Reverse) gradients;
# - starts: WarmupHMC's default `init` (a Uniform(-2, 2) draw from each chain's
#   RNG, then Pathfinder from it); the wrapper also starts from its default
#   controls, every cell non-centred (`c = 0`);
# - reported: max R-hat and minimum bulk/tail ESS over every coordinate of the
#   kept chains (MCMCDiagnosticTools `ess_rhat`), the same over the intercept
#   block and the population intercept, and divergences.
#
# In one environment a re-run repeats every printed figure (two runs at BRM
# 0.2.312 on gordito). Across environments they move with the BRM, WarmupHMC,
# ReactiveKernels and Enzyme builds, so compare another environment or language
# against a tolerance, not the digits. brm-use `references/centering.md`
# ("Adaptive partial centering of an RKBRMI") lists the figures per environment.
#
#     julia --project=test research/adaptive_centering/rk_panel.jl [seeds...]
using BayesianRegressionModels, Distributions, Random, LinearAlgebra, Statistics
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems, WarmupHMC
using DifferentiationInterface: AutoEnzyme
using CategoricalArrays, MCMCDiagnosticTools
const BRM = BayesianRegressionModels
const AD = AutoEnzyme(; mode=Enzyme.Reverse)

function panel(; G=12, n=30, seed=1)
    rng = Xoshiro(seed)
    level = 60 .+ 70 .* rand(rng, G)
    slope = 0.5 .* randn(rng, G, 3)
    g = repeat(1:G; inner=n)
    t = repeat(collect(range(-1.0, 1.0; length=n)), G)
    c1 = randn(rng, G * n); c2 = randn(rng, G * n)
    se = 0.3 .+ 0.3 .* rand(rng, G * n)
    y = level[g] .+ (1.0 .+ slope[g, 1]) .* t .+ (0.5 .+ slope[g, 2]) .* c1 .+
        (-0.3 .+ slope[g, 3]) .* c2 .+ se .* randn(rng, G * n)
    (; y, t, c1, c2, se, g=categorical(string.("g", lpad.(g, 2, '0'))))
end

model(df) = @brm df begin
    mu ~ 1 + t + c1 + c2 + (1 + t + c1 + c2 || g)
    sd(mu, g, Intercept) ~ Exponential(30.0)
    sd(mu, g, t) ~ Exponential(1.0)
    sd(mu, g, c1) ~ Exponential(1.0)
    sd(mu, g, c2) ~ Exponential(1.0)
    y ~ Normal(mu, se)
end

function summarize(label, out, names, focus)
    chains = [Matrix(r.posterior_position) for r in out.results]
    A = cat((permutedims(c) for c in chains)...; dims=3)      # draws x params x chains
    A = permutedims(A, (1, 3, 2))                              # draws x chains x params
    ess, rhat = ess_rhat(A)
    tail = ess_rhat(A; kind=:tail)[1]
    sel = findall(n -> any(f -> occursin(f, n), focus), names)
    worst = sortperm(rhat; rev=true)[1:5]
    ndiv = out.completion.n_divergent_samples
    println(rpad(label, 26),
        " kept=", out.completion.n_completed, "/", out.completion.n_requested,
        " maxRhat=", round(maximum(rhat); digits=3),
        " minBulkESS=", round(minimum(ess); digits=1),
        " minTailESS=", round(minimum(tail); digits=1),
        " intercept-block maxRhat=", round(maximum(rhat[sel]); digits=3),
        " minESS=", round(minimum(ess[sel]); digits=1),
        " div=", ndiv, " (", round(100ndiv / out.completion.n_samples; digits=2), "%)",
        " secs=", round(out.completion.elapsed_seconds; digits=1))
    for j in worst
        println("    R-hat ", round(rhat[j]; digits=3), " ESS ", round(ess[j]; digits=1),
                "  ", names[j])
    end
    flush(stdout)
end

seeds = isempty(ARGS) ? [51, 52] : parse.(Int, ARGS)
df = panel(; seed=1)
rk = RKBRMI(model(df))
names = String.(BRM._rk_layout_coordinate_names(rk))
problem = rk_logdensity_problem(rk; ad_backend=AD)
intercept_draws = String(only(b.ranef.binding for b in adaptive_centering_blocks(rk)
    if only(BRM._rk_adaptive_margins(rk)[b.ranef.binding]).coefficient === :Intercept))
focus = [intercept_draws * ".", "pop_mu.beta_pop.1"]
println("Julia ", VERSION, "; threads ", Threads.nthreads(), "; dim ", length(names),
        "; intercept block ", intercept_draws)
for seed in seeds
    rngs() = [Xoshiro(1000seed + i) for i in 1:4]
    plain = completion_warmup_mcmc(rngs(), problem; min_completed=3, n_draws=400)
    summarize("seed $seed plain RK", plain, names, focus)
    adaptive = adaptive_centering_problem(rk, problem, AD)
    wrapped = completion_warmup_mcmc(rngs(), adaptive; min_completed=3, n_draws=400)
    summarize("seed $seed adaptive RK", wrapped, names, focus)
end
